#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <thread>
#include <iostream>

/**
 The parallel construct creates a team of OpenMP threads that executes the
structured block followed by parallel region.

Scope
    shared: the data declared outside a parallel region is shared, 
            which means visible and accessible by all threads simultaneously.
            By default, all variables in the work sharing region are shared except 
            the loop iteration counter.
    private: the data declared within a parallel region is private to each thread,
             which means each thread will have a local copy and use it as a temporary variable. 
             A private variable is not initialized and the value is not maintained for use 
             outside the parallel region.
    Use the below clauses to restrict the scope of a variable.       
    shared(list) -> makes the variables are shared
    private(list) -> specifies that each thread have its own instance of avariable/variables
    firstprivate(list) -> same as private, except that it initialized with the variables value.

shared: the data declared outside a parallel region is shared, which means visible and accessible by all threads simultaneously. By default, all variables in the work sharing region are shared except the loop iteration counter.

Condition:
    if -> A scalar expression to specify whether a parallel region should be
executed in parallel or in serial.

Affinity & Placement:
    Affinity -> Assigns a preference for the scheduling of a process,
                rank or thread to a particular hardware component.
                This is also called pinning or binding.

    Placement -> Assigns a process or thread to a hardware location.

    OMP_PLACES and OMP_PROC_BIND environment variables to specify how the OpenMP
threads in a program are bound to processors. These two environment variable are
often used in conjunction with each other. OMP_PLACES is used to specify the
places on the machine to which the threads are bound. OMP_PROC_BIND is used to
specify the binding policy (thread affinity policy) which prescribes how the
threads are assigned to places. Setting OMP_PLACES alone does not enable
binding. You also need to set OMP_PROC_BIND.

    The value of OMP_PLACES can be one of two types of values:
        1. Abstract name [threads|cores|sockets].
                threads: a place is a single hardware thread
                cores: a place is a single core with its corresponding amount of
hardware threads sockets: a place is a single socket
        2. An explicit list of places described by non-negative numbers.
                In order to define specific places by an interval, OMP_PLACES
                can be set to <lowerbound>:<length>:<stride>

    OMP_PROC_BIND[true, false, close, spread, primary]
        These will tell if the threads to be spread or close
*/
static void parallel_region() {
    bool executeInParallel = true;
    int array[]{19, 2, 3}; // static array
    int *arrayPtr = new int[10]; // dynamic array
    for (int i = 0; i < 10; i++) {
        arrayPtr[i] = 100 + i;
    }
    #pragma omp parallel if (executeInParallel) firstprivate(arrayPtr)
    {
        #pragma omp critical 
        {
            std::cout << "Thread id: " << omp_get_thread_num() << " Address: " << arrayPtr << std::endl;
        }
    }  // implicit barrier.
}

int main(int argc, char const *argv[]) {
    parallel_region();
    return 0;
}
