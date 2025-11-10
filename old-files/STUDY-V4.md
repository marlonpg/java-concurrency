# Java Concurrency Deep Dive Presentation

## Table of Contents
1. [Introduction to Concurrency](#1-introduction-to-concurrency)
2. [Thread Fundamentals](#2-thread-fundamentals)
3. [Synchronization Mechanisms](#3-synchronization-mechanisms)
4. [Java Memory Model](#4-java-memory-model)
5. [Concurrent Collections](#5-concurrent-collections)
6. [Executor Framework](#6-executor-framework)
7. [Advanced Concurrency Utilities](#7-advanced-concurrency-utilities)
8. [Common Concurrency Patterns](#8-common-concurrency-patterns)
9. [Performance Considerations](#9-performance-considerations)
10. [Best Practices and Pitfalls](#10-best-practices-and-pitfalls)

---

## 1. Introduction to Concurrency

### What is Concurrency?
- **Definition**: Ability to execute multiple tasks simultaneously
- **Parallelism vs Concurrency**: 
  - Concurrency: Dealing with multiple things at once (composition)
  - Parallelism: Doing multiple things at once (execution)

### Why Concurrency Matters
- **Performance**: Utilize multiple CPU cores
- **Responsiveness**: Keep UI responsive while processing
- **Throughput**: Handle more requests simultaneously
- **Resource Utilization**: Better use of system resources

### Challenges of Concurrent Programming
- **Race Conditions**: Multiple threads accessing shared data
- **Deadlocks**: Circular dependency between threads
- **Starvation**: Thread never gets CPU time
- **Livelock**: Threads keep changing state but make no progress

---

## 2. Thread Fundamentals

### Thread Lifecycle
```
NEW → RUNNABLE → BLOCKED/WAITING/TIMED_WAITING → TERMINATED
```

### Creating Threads
1. **Extending Thread class**
2. **Implementing Runnable interface**
3. **Using Lambda expressions**
4. **Callable interface (with return value)**

### Thread Methods
- `start()` vs `run()`
- `join()` - Wait for thread completion
- `interrupt()` - Interrupt thread execution
- `sleep()` - Pause execution
- `yield()` - Hint to scheduler

### Daemon Threads
- Background threads that don't prevent JVM shutdown
- Garbage collector, finalizer threads
- `setDaemon(true)` before starting

---

## 3. Synchronization Mechanisms

### The `synchronized` Keyword
- **Method synchronization**: `synchronized void method()`
- **Block synchronization**: `synchronized(object) { }`
- **Static synchronization**: Class-level locking

### Intrinsic Locks (Monitor Locks)
- Every object has an intrinsic lock
- Reentrant nature
- `wait()`, `notify()`, `notifyAll()`

### Volatile Keyword
- Ensures visibility of changes across threads
- Prevents instruction reordering
- Not atomic for compound operations

### Atomic Classes
- `AtomicInteger`, `AtomicLong`, `AtomicReference`
- Compare-and-swap (CAS) operations
- Lock-free programming

### Explicit Locks
- `ReentrantLock`: More flexible than synchronized
- `ReadWriteLock`: Separate read/write locks
- `StampedLock`: Optimistic reading (Java 8+)

---

## 4. Java Memory Model (JMM)

### Memory Visibility
- **Happens-before relationship**
- **Memory barriers/fences**
- **Cache coherence**

### Key Concepts
- **Main Memory vs Working Memory**
- **Volatile variables guarantee**
- **Synchronization guarantees**

### Memory Consistency Errors
- **Stale data**
- **Lost updates**
- **Inconsistent state**

### JMM Rules
1. Program order rule
2. Monitor lock rule
3. Volatile variable rule
4. Thread start rule
5. Thread termination rule
6. Interruption rule
7. Finalizer rule
8. Transitivity

---

## 5. Concurrent Collections

### Thread-Safe Collections
- `Vector`, `Hashtable` (legacy, synchronized)
- `Collections.synchronizedXxx()` wrappers

### Concurrent Collections (java.util.concurrent)
- **ConcurrentHashMap**: Segment-based locking
- **CopyOnWriteArrayList**: Copy-on-write semantics
- **ConcurrentLinkedQueue**: Lock-free queue
- **BlockingQueue implementations**:
  - `ArrayBlockingQueue`
  - `LinkedBlockingQueue`
  - `PriorityBlockingQueue`
  - `DelayQueue`
  - `SynchronousQueue`

### Performance Characteristics
- Lock contention vs throughput
- Memory overhead
- Iteration consistency

---

## 6. Executor Framework

### Problems with Manual Thread Management
- Thread creation overhead
- Resource management
- Exception handling
- Thread lifecycle management

### Executor Interface Hierarchy
```
Executor → ExecutorService → ScheduledExecutorService
```

### Thread Pool Types
- **FixedThreadPool**: Fixed number of threads
- **CachedThreadPool**: Creates threads as needed
- **SingleThreadExecutor**: Single worker thread
- **ScheduledThreadPool**: Delayed/periodic execution
- **WorkStealingPool**: Fork-Join based (Java 8+)

### Future and CompletableFuture
- Asynchronous computation results
- `get()`, `cancel()`, `isDone()`
- CompletableFuture: Composable async programming

### Best Practices
- Always shutdown executors
- Handle RejectedExecutionException
- Monitor thread pool metrics

---

## 7. Advanced Concurrency Utilities

### CountDownLatch
- One-time synchronization barrier
- Wait for multiple threads to complete
- Cannot be reset

### CyclicBarrier
- Reusable synchronization barrier
- All threads wait at barrier point
- Optional barrier action

### Semaphore
- Controls access to resource pool
- Permits-based access control
- Fair vs non-fair acquisition

### Exchanger
- Synchronous data exchange between two threads
- Bidirectional synchronization point

### Phaser (Java 7+)
- Flexible synchronization barrier
- Dynamic thread registration
- Multi-phase synchronization

### Fork-Join Framework
- Divide-and-conquer parallelism
- Work-stealing algorithm
- `ForkJoinPool`, `RecursiveTask`, `RecursiveAction`

---

## 8. Common Concurrency Patterns

### Producer-Consumer Pattern
- BlockingQueue implementation
- Wait-notify mechanism
- Bounded buffer problem

### Reader-Writer Pattern
- ReadWriteLock usage
- Multiple readers, single writer
- Starvation prevention

### Thread Pool Pattern
- Task submission and execution
- Resource pooling
- Load balancing

### Future Pattern
- Asynchronous result retrieval
- Non-blocking operations
- Callback mechanisms

### Actor Model
- Message-passing concurrency
- Isolated state
- Akka framework example

---

## 9. Performance Considerations

### Measuring Concurrency Performance
- **Throughput**: Tasks per unit time
- **Latency**: Time to complete single task
- **Scalability**: Performance with increased load
- **Resource utilization**: CPU, memory usage

### Performance Bottlenecks
- **Lock contention**: Too many threads competing
- **Context switching**: Overhead of thread switching
- **False sharing**: Cache line contention
- **Memory allocation**: GC pressure

### Optimization Techniques
- **Lock-free algorithms**: CAS-based operations
- **Thread-local storage**: Avoid sharing
- **Batching**: Reduce synchronization frequency
- **Partitioning**: Divide work to reduce contention

### Profiling Tools
- JProfiler, YourKit
- JVM built-in tools (jstack, jstat)
- Application Performance Monitoring (APM)

---

## 10. Best Practices and Pitfalls

### Best Practices
1. **Prefer immutable objects**
2. **Use thread-safe collections**
3. **Minimize shared mutable state**
4. **Use higher-level concurrency utilities**
5. **Document thread safety guarantees**
6. **Handle interruption properly**
7. **Avoid thread leaks**
8. **Use timeouts for blocking operations**

### Common Pitfalls
1. **Double-checked locking** (without volatile)
2. **Inconsistent synchronization**
3. **Deadlock scenarios**
4. **Race conditions in initialization**
5. **Improper exception handling in threads**
6. **Memory leaks with ThreadLocal**
7. **Blocking operations in critical sections**

### Code Review Checklist
- [ ] Thread safety documented
- [ ] Proper synchronization used
- [ ] No race conditions
- [ ] Deadlock prevention
- [ ] Resource cleanup
- [ ] Exception handling
- [ ] Performance considerations

---

## Demo Code Examples

### Basic Thread Creation
```java
// Runnable interface
Thread thread = new Thread(() -> {
    System.out.println("Running in: " + Thread.currentThread().getName());
});
thread.start();

// Callable with Future
ExecutorService executor = Executors.newSingleThreadExecutor();
Future<String> future = executor.submit(() -> "Hello from Callable");
String result = future.get();
```

### Synchronization Example
```java
public class Counter {
    private int count = 0;
    
    public synchronized void increment() {
        count++;
    }
    
    public synchronized int getCount() {
        return count;
    }
}
```

### Producer-Consumer with BlockingQueue
```java
BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);

// Producer
new Thread(() -> {
    try {
        queue.put("item");
    } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
    }
}).start();

// Consumer
new Thread(() -> {
    try {
        String item = queue.take();
        System.out.println("Consumed: " + item);
    } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
    }
}).start();
```

---

## Q&A Preparation

### Common Interview Questions
1. What's the difference between `synchronized` and `ReentrantLock`?
2. How does `ConcurrentHashMap` work internally?
3. What is the Java Memory Model?
4. When would you use `CountDownLatch` vs `CyclicBarrier`?
5. How do you handle deadlocks?
6. What's the difference between `submit()` and `execute()`?

### Advanced Topics for Discussion
- Lock-free programming
- Memory models in different architectures
- Reactive programming with CompletableFuture
- Virtual threads (Project Loom)
- Structured concurrency

---

## Resources for Further Learning
- "Java Concurrency in Practice" by Brian Goetz
- "Effective Java" by Joshua Bloch
- Oracle Java Documentation
- OpenJDK Concurrency JSRs
- Doug Lea's concurrent programming papers