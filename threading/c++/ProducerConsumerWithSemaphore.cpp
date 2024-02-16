#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <semaphore>

template <typename T>
class BlockingQueue_Semaphore {
   private:
    int capacity;
    T* buffer;
    int size = 0;
    int head = 0;
    int tail = 0;
    std::counting_semaphore<10> space{3};
    std::counting_semaphore<10> items {0};
    std::mutex mutex;

   public:
    BlockingQueue_Semaphore(int capacity) : capacity(capacity) {
        buffer = new T[capacity];
    }

    void enqueue(T elem) {
        // check if you can get a permit to write.
        space.acquire();
        mutex.lock();
            if (tail == capacity) {
                tail = 0;
            }
            buffer[tail] = elem;
            size++;
            tail++;
        mutex.unlock();
        // signal to the consumer to consume mesages
        items.release();
    }

    T dequeue() {
        // check if you can get a permit to read from buffer.
        items.acquire();
        mutex.lock();
            if (head == capacity) {
                head = 0;
            }
            T elem = buffer[head];
            head++;
            size--;
        mutex.unlock();
        space.release();
        return elem;
    }
};

template <typename T>
void consumeMessages(BlockingQueue_Semaphore<T>& queue) {
    while (true) {
        T elem = queue.dequeue();
        std::cout << "Consumed: " << elem << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

template <typename T>
void publishMessages(BlockingQueue_Semaphore<T>& queue) {
    while (true) {
        std::string message("My Message");
        queue.enqueue(message);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

int main(int argc, char const* argv[]) {
    BlockingQueue_Semaphore<std::string> blockingQueue(3);
    std::thread publisher(publishMessages<std::string>, std::ref(blockingQueue)); 
    std::thread consumer(consumeMessages<std::string>, std::ref(blockingQueue)); 
    publisher.join();
    consumer.join();
    return 0;
};
