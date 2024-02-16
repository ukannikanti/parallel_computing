#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

class WithStarvation {
   private:
    int readers = 0;
    std::mutex mutex;  // to lock the readers variable
    std::counting_semaphore<10> roomEmpty{1};
    // to track if the critical section is occupied by readers/writers

   public:
    void readersFunc() {
        // Entering readers
        mutex.lock();
        readers = readers + 1;
        if (readers == 1)
            // For the first reader thread get a permit to enter the critical
            // block
            roomEmpty.acquire();
        mutex.unlock();

        std::cout << "Readers entered the critical section!!" << std::endl;

        // exiting readers
        mutex.lock();
        readers = readers - 1;
        if (readers == 0)
            // For the last reader, signal the other threads that room is empty.
            roomEmpty.release();
        mutex.unlock();
    }

    void writers() {
        // Get a permit to enter the critical section to write some stuff..
        roomEmpty.acquire();

        std::cout << "Writers entered the critical section!!" << std::endl;

        roomEmpty.release();
    }
};

class NoStarvation {
   private:
    int readers = 0;
    std::mutex mutex;
    std::counting_semaphore<10> turnstile{1};
    std::counting_semaphore<10> roomEmpty{1};

   public:
    void readersFunc() {
        turnstile.acquire();
        turnstile.release();
        mutex.lock();
        readers = readers + 1;
        if (readers == 1) {
            roomEmpty.acquire();
        }
        mutex.unlock();

        std::cout << "Readers entered the critical section!!" << std::endl;

        mutex.lock();
        readers = readers - 1;
        if (readers == 0) {
            roomEmpty.release();
        }
        mutex.unlock();
    }

    void writers() {
        turnstile.acquire();
        roomEmpty.acquire();
        std::cout << "Writers entered the critical section!!" << std::endl;
        turnstile.release();
        roomEmpty.release();
    }
};

class WritersPriority {
   private:
    int writersCount;
    int readers;
    std::mutex readMutex;
    std::mutex writeMutex;
    std::counting_semaphore<10> noReaders{1};
    std::counting_semaphore<10> noWriters{1};

   public:
    void readersFunc() {
        noReaders.acquire();
            readMutex.lock();
                readers = readers + 1;
                if (readers == 1) {
                    noWriters.acquire();  // Ensure there are no writers in the critical section
                }
            readMutex.unlock();
        noReaders.release();

        std::cout << "Readers entered the critical section!!" << std::endl;

        readMutex.lock();
        readers = readers - 1;
        if (readers == 0) {
            noWriters.release();
        }
        readMutex.unlock();
    }

    void writers() {
        writeMutex.lock();
        writersCount = writersCount + 1;
        if (writersCount == 1) {
            noReaders.acquire();  // Wait for there are no readers
        }
        writeMutex.unlock();

        noWriters.acquire();
        std::cout << "Writers!! entered the critical section!!" << std::endl;
        noWriters.release();

        writeMutex.lock();
        writersCount = writersCount - 1;
        if (writersCount == 0) {
            // signal the last writer to write
            if (writersCount == 0) {
                noReaders.release();
            }
        }
        writeMutex.unlock();
    }
};

template <typename T>
void writers(T& obj) {
    while (true) {
        obj.writers();
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
}

template <typename T>
void readers(T& obj) {
    while (true) {
        obj.readersFunc();
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void executeWritersStarvation() {
    WithStarvation obj;
    std::thread reader1(readers<WithStarvation>, std::ref(obj));
    std::thread reader2(readers<WithStarvation>, std::ref(obj));
    std::thread reader3(readers<WithStarvation>, std::ref(obj));

    std::thread writersT(writers<WithStarvation>, std::ref(obj));
    reader1.join();
    reader2.join();
    reader3.join();
    writersT.join();
}

void executeNoStarvation() {
    NoStarvation obj;
    std::thread reader1(readers<NoStarvation>, std::ref(obj));
    std::thread reader2(readers<NoStarvation>, std::ref(obj));
    std::thread reader3(readers<NoStarvation>, std::ref(obj));
    std::thread reader4(readers<NoStarvation>, std::ref(obj));
    std::thread reader5(readers<NoStarvation>, std::ref(obj));
    std::thread writer1(writers<NoStarvation>, std::ref(obj));
    std::thread writer2(writers<NoStarvation>, std::ref(obj));
    reader1.join();
    reader2.join();
    reader3.join();
    reader4.join();
    reader5.join();
    writer1.join();
    writer2.join();
}

void executeWritePriority() {
    WritersPriority obj;
    std::thread reader1(readers<WritersPriority>, std::ref(obj));
    std::thread reader2(readers<WritersPriority>, std::ref(obj));
    std::thread reader3(readers<WritersPriority>, std::ref(obj));
    std::thread reader4(readers<WritersPriority>, std::ref(obj));
    std::thread reader5(readers<WritersPriority>, std::ref(obj));
    std::thread reader6(readers<WritersPriority>, std::ref(obj));
    std::thread reader7(readers<WritersPriority>, std::ref(obj));
    std::thread writer1(writers<WritersPriority>, std::ref(obj));
    std::thread writer2(writers<WritersPriority>, std::ref(obj));
    reader1.join();
    reader2.join();
    reader3.join();
    reader4.join();
    reader5.join();
    reader6.join();
    reader7.join();
    writer1.join();
    writer2.join();
}

int main(int argc, char const* argv[]) {
    executeWritePriority();
    return 0;
};