#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

class Philosophers {
   private:
    // Only 4 philospers allowed at a time to avoid deadlock.
    std::counting_semaphore<10> permits{4};
    // forks are shared among the philospers to eat and there's only 5
    std::mutex forks[5];
    std::mutex coutMutex;
    std::mutex coutMutex1;

   public:
    void eat(int id) {
        coutMutex1.lock();
            std::cout << "Philospher: " << id << " eating" << std::endl;
        coutMutex1.unlock();
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        coutMutex1.lock();
            std::cout << "Philospher: " << id << " done eating" << std::endl;
        coutMutex1.unlock();
    }

    void think(int id) {
        coutMutex.lock();
            std::cout << "Philospher: " << id << " thinking" << std::endl;
        coutMutex.unlock();
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        coutMutex.lock();
            std::cout << "Philospher: " << id << " done thinking" << std::endl;
         coutMutex.unlock();
    }

    void getForks(int philospher) {
        permits.acquire();
        int left = philospher;
        int right = (philospher + 1) % 5;
        forks[left].lock();
        forks[right].lock();
    }

    void putForks(int philospher) {
        int left = philospher;
        int right = (philospher + 1) % 5;
        forks[left].unlock();
        forks[right].unlock();
        permits.release();
    }

    void philospher(int id) {
        think(id);
        getForks(id);
        eat(id);
        putForks(id);
    }

    void run() {
        // 5 threads to represent 5 philosphers
        std::thread philosphers[5];
        for (int i = 0; i < 5; i++) {
            philosphers[i] = std::thread([&, i]() { philospher(i);});
        }

        for (std::thread& t : philosphers) {
            t.join();
        }
    }
};

int main(int argc, char const* argv[]) {
    Philosophers p;
    p.run();
    return 0;
}
