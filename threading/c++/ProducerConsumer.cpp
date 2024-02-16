#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

template <typename T>
class BlockingQueue {
   private:
    int capacity;
    T* buffer;
    int size = 0;
    int head = 0;
    int tail = 0;

    std::mutex pMutex;
    std::mutex cMutex;
    std::condition_variable cv;

   public:
    BlockingQueue(int capacity) : capacity(capacity) {
        buffer = new T[capacity];
    }

    void enqueue(T elem) {
        std::unique_lock<std::mutex> lock(pMutex);
        cv.wait(lock, [this](){return size != capacity;});

        if (tail == capacity) {
            tail = 0;
        }
        
        buffer[tail] = elem;
        std::cout << "Added elem" << elem << std::endl;
        size++;
        tail++;
        cv.notify_all();
    }

    T dequeue() {
        std::unique_lock<std::mutex> lock(cMutex);
        cv.wait(lock, [this](){return tail != 0;});

        if (head == capacity) {
            head = 0;
        }

        T elem = buffer[head];
        head++;
        size--;
        cv.notify_all();
        return elem;
    }
};

template <typename T>
void publishMessages(BlockingQueue<T>& queue) {
    while (true) {
        std::string message("My Message");
        std::cout << "Added: " << message << std::endl;
        queue.enqueue(message);
        std::cout << "Added: " << message << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

template <typename T>
void consumeMessages(BlockingQueue<T>& queue) {
    while (true) {
        T elem = queue.dequeue();
        std::cout << "Consumed: " << elem << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

int main(int argc, char const* argv[]) {
    BlockingQueue<std::string> blockingQueue(3);
    std::thread publisher(publishMessages<std::string>, std::ref(blockingQueue)); 
    std::thread consumer(consumeMessages<std::string>, std::ref(blockingQueue)); 
    publisher.join();
    consumer.join();
    return 0;
};