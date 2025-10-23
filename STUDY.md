# Java Concurrency Deep Study Guide

## 1. Thread Fundamentals - Deep Dive

### Thread States and Transitions

```java
public enum Thread.State {
    NEW,           // Thread created but not started
    RUNNABLE,      // Executing or ready to execute
    BLOCKED,       // Blocked waiting for monitor lock
    WAITING,       // Waiting indefinitely for another thread
    TIMED_WAITING, // Waiting for specified period
    TERMINATED     // Thread has completed execution
}
```

**Key Understanding:**
- `NEW`: Thread object created but `start()` not called
- `RUNNABLE`: Thread may be running or waiting for CPU time
- `BLOCKED`: Waiting to acquire a synchronized block/method
- `WAITING`: Called `wait()`, `join()`, or `LockSupport.park()`
- `TIMED_WAITING`: Called `sleep()`, `wait(timeout)`, `join(timeout)`
- `TERMINATED`: `run()` method completed or exception thrown

### Thread Creation Methods

```java
// Method 1: Extending Thread
class MyThread extends Thread {
    @Override
    public void run() {
        System.out.println("Thread: " + getName());
    }
}

// Method 2: Implementing Runnable (Preferred)
class MyTask implements Runnable {
    @Override
    public void run() {
        System.out.println("Task running");
    }
}

// Method 3: Lambda expression
Thread t = new Thread(() -> System.out.println("Lambda thread"));

// Method 4: Callable (returns value)
Callable<String> task = () -> "Result from callable";
```

**Why Runnable is preferred:**
- Java single inheritance limitation
- Better separation of concerns
- Can be used with Executor framework

### Critical Thread Methods

```java
// start() vs run()
Thread t = new Thread(() -> System.out.println("Hello"));
t.run();   // Executes in current thread (wrong!)
t.start(); // Creates new thread and calls run() (correct!)

// join() - Wait for thread completion
Thread worker = new Thread(() -> {
    try { Thread.sleep(1000); } catch (InterruptedException e) {}
});
worker.start();
worker.join(); // Main thread waits for worker to complete

// interrupt() - Cooperative cancellation
Thread t = new Thread(() -> {
    while (!Thread.currentThread().isInterrupted()) {
        // Do work
        if (Thread.interrupted()) { // Clears interrupt flag
            break;
        }
    }
});
t.interrupt(); // Sets interrupt flag
```

**Interrupt Handling Best Practice:**
```java
public void interruptibleMethod() throws InterruptedException {
    while (!Thread.currentThread().isInterrupted()) {
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt(); // Restore interrupt status
            throw e; // Re-throw to caller
        }
    }
}
```

---

## 2. Synchronization Mechanisms - Deep Dive

### The synchronized Keyword

**Method-level synchronization:**
```java
public class Counter {
    private int count = 0;
    
    // Equivalent to synchronized(this)
    public synchronized void increment() {
        count++; // Not atomic without synchronization
    }
    
    // Static synchronization uses Class object
    public static synchronized void staticMethod() {
        // synchronized(Counter.class)
    }
}
```

**Block-level synchronization:**
```java
public class BankAccount {
    private double balance;
    private final Object lock = new Object(); // Private lock object
    
    public void withdraw(double amount) {
        synchronized(lock) { // Better than synchronized(this)
            if (balance >= amount) {
                balance -= amount;
            }
        }
    }
}
```

**Why private lock objects are better:**
- Prevents external code from acquiring your lock
- Avoids accidental deadlocks
- Better encapsulation

### Intrinsic Locks and Monitor Pattern

```java
public class ProducerConsumer {
    private final Object lock = new Object();
    private Queue<String> queue = new LinkedList<>();
    private final int MAX_SIZE = 10;
    
    public void produce(String item) throws InterruptedException {
        synchronized(lock) {
            while (queue.size() == MAX_SIZE) {
                lock.wait(); // Releases lock and waits
            }
            queue.offer(item);
            lock.notifyAll(); // Wake up all waiting threads
        }
    }
    
    public String consume() throws InterruptedException {
        synchronized(lock) {
            while (queue.isEmpty()) {
                lock.wait();
            }
            String item = queue.poll();
            lock.notifyAll();
            return item;
        }
    }
}
```

**Key Points:**
- `wait()` must be called inside synchronized block
- Always use `while` loop, not `if` (spurious wakeups)
- `notifyAll()` is safer than `notify()`

### Volatile Keyword Deep Dive

```java
public class VolatileExample {
    private volatile boolean flag = false;
    private int counter = 0;
    
    public void writer() {
        counter = 42;    // Happens-before
        flag = true;     // Volatile write
    }
    
    public void reader() {
        if (flag) {      // Volatile read
            // counter is guaranteed to be 42 here
            System.out.println(counter);
        }
    }
}
```

**Volatile guarantees:**
1. **Visibility**: Changes visible to all threads immediately
2. **Ordering**: Prevents reordering around volatile operations
3. **Not atomic**: `volatile int i; i++` is still not thread-safe

**When to use volatile:**
- Simple flags or status variables
- Double-checked locking pattern (with proper implementation)
- Publisher-subscriber scenarios

### Atomic Classes

```java
public class AtomicCounter {
    private final AtomicInteger count = new AtomicInteger(0);
    
    public int increment() {
        return count.incrementAndGet(); // Atomic operation
    }
    
    public boolean compareAndSet(int expected, int update) {
        return count.compareAndSet(expected, update);
    }
    
    // Custom atomic operation
    public int addTen() {
        return count.updateAndGet(current -> current + 10);
    }
}
```

**Compare-and-Swap (CAS) Algorithm:**
```java
// Conceptual implementation of CAS
public boolean compareAndSwap(int expected, int newValue) {
    if (currentValue == expected) {
        currentValue = newValue;
        return true;
    }
    return false;
}
```

### Explicit Locks

```java
public class ReentrantLockExample {
    private final ReentrantLock lock = new ReentrantLock();
    private int count = 0;
    
    public void increment() {
        lock.lock();
        try {
            count++;
        } finally {
            lock.unlock(); // Always in finally block
        }
    }
    
    public boolean tryIncrement() {
        if (lock.tryLock()) {
            try {
                count++;
                return true;
            } finally {
                lock.unlock();
            }
        }
        return false;
    }
    
    public boolean tryIncrementWithTimeout() throws InterruptedException {
        if (lock.tryLock(1, TimeUnit.SECONDS)) {
            try {
                count++;
                return true;
            } finally {
                lock.unlock();
            }
        }
        return false;
    }
}
```

**ReentrantLock vs synchronized:**
- **Flexibility**: tryLock(), timed locking, interruptible locking
- **Fairness**: Can be fair (FIFO) or unfair
- **Condition variables**: Multiple wait sets
- **Performance**: Similar in modern JVMs
- **Complexity**: More complex, must remember to unlock

### ReadWriteLock

```java
public class ReadWriteCache<K, V> {
    private final Map<K, V> cache = new HashMap<>();
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private final Lock readLock = lock.readLock();
    private final Lock writeLock = lock.writeLock();
    
    public V get(K key) {
        readLock.lock();
        try {
            return cache.get(key);
        } finally {
            readLock.unlock();
        }
    }
    
    public void put(K key, V value) {
        writeLock.lock();
        try {
            cache.put(key, value);
        } finally {
            writeLock.unlock();
        }
    }
}
```

---

## 3. Java Memory Model (JMM) - Deep Dive

### Happens-Before Relationship

**Definition**: If action A happens-before action B, then the memory effects of A are visible to B.

**Happens-Before Rules:**
1. **Program Order**: Each action happens-before every subsequent action in the same thread
2. **Monitor Lock**: Unlock happens-before every subsequent lock on the same monitor
3. **Volatile**: Write to volatile field happens-before every subsequent read of that field
4. **Thread Start**: `Thread.start()` happens-before any action in the started thread
5. **Thread Termination**: Any action in thread happens-before `join()` returns
6. **Interruption**: `interrupt()` happens-before interrupted thread detects interruption
7. **Finalizer**: Constructor completion happens-before finalizer starts
8. **Transitivity**: If A happens-before B and B happens-before C, then A happens-before C

### Memory Visibility Example

```java
public class MemoryVisibilityExample {
    private int data = 0;
    private volatile boolean ready = false;
    
    // Thread 1
    public void writer() {
        data = 42;        // 1
        ready = true;     // 2 (volatile write)
    }
    
    // Thread 2
    public void reader() {
        if (ready) {      // 3 (volatile read)
            // data is guaranteed to be 42 due to happens-before
            assert data == 42; // This will never fail
        }
    }
}
```

**Explanation**: Volatile write (2) happens-before volatile read (3), and due to program order, (1) happens-before (2), so by transitivity, (1) happens-before (3).

### Double-Checked Locking Pattern

**Broken implementation:**
```java
public class Singleton {
    private static Singleton instance;
    
    public static Singleton getInstance() {
        if (instance == null) {           // Check 1
            synchronized(Singleton.class) {
                if (instance == null) {   // Check 2
                    instance = new Singleton(); // Problem: not atomic!
                }
            }
        }
        return instance;
    }
}
```

**Correct implementation:**
```java
public class Singleton {
    private static volatile Singleton instance; // volatile is crucial
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized(Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**Why volatile is needed**: Object construction involves multiple steps:
1. Allocate memory
2. Initialize object
3. Assign reference

Without volatile, steps 2 and 3 can be reordered, leading to partially constructed objects.

---

## 4. Concurrent Collections - Deep Dive

### ConcurrentHashMap Internal Structure

**Java 7 - Segment-based:**
```java
// Conceptual structure
class ConcurrentHashMap<K,V> {
    final Segment<K,V>[] segments; // Array of segments
    
    static class Segment<K,V> extends ReentrantLock {
        volatile HashEntry<K,V>[] table; // Hash table per segment
    }
}
```

**Java 8+ - Node-based with CAS:**
```java
// Simplified structure
class ConcurrentHashMap<K,V> {
    volatile Node<K,V>[] table;
    
    // Uses CAS for updates, synchronized only for tree operations
    final V putVal(K key, V value, boolean onlyIfAbsent) {
        // CAS-based insertion with fallback to synchronized
    }
}
```

**Key improvements in Java 8+:**
- Better scalability (no segment limit)
- CAS-based operations for better performance
- Tree structure for hash collision handling

### BlockingQueue Implementations

```java
// ArrayBlockingQueue - bounded, array-based
BlockingQueue<String> bounded = new ArrayBlockingQueue<>(100);

// LinkedBlockingQueue - optionally bounded, linked-list based
BlockingQueue<String> unbounded = new LinkedBlockingQueue<>();
BlockingQueue<String> bounded2 = new LinkedBlockingQueue<>(100);

// PriorityBlockingQueue - unbounded, heap-based
BlockingQueue<Task> priority = new PriorityBlockingQueue<>();

// SynchronousQueue - no storage, direct handoff
BlockingQueue<String> handoff = new SynchronousQueue<>();

// DelayQueue - elements available after delay
BlockingQueue<DelayedTask> delayed = new DelayQueue<>();
```

**Producer-Consumer with different queues:**
```java
public class ProducerConsumerExample {
    private final BlockingQueue<Integer> queue;
    
    public ProducerConsumerExample(BlockingQueue<Integer> queue) {
        this.queue = queue;
    }
    
    public void producer() {
        try {
            for (int i = 0; i < 100; i++) {
                queue.put(i); // Blocks if queue is full
                Thread.sleep(10);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    public void consumer() {
        try {
            while (!Thread.currentThread().isInterrupted()) {
                Integer item = queue.take(); // Blocks if queue is empty
                processItem(item);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

### CopyOnWriteArrayList

```java
public class CopyOnWriteExample {
    private final CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();
    
    public void writer() {
        list.add("item"); // Creates new array copy
    }
    
    public void reader() {
        // Iterator uses snapshot, won't see concurrent modifications
        for (String item : list) {
            System.out.println(item);
        }
    }
}
```

**When to use CopyOnWriteArrayList:**
- Read operations vastly outnumber write operations
- List size is relatively small
- Iteration consistency is important

---

## 5. Executor Framework - Deep Dive

### Thread Pool Types and Characteristics

```java
// Fixed thread pool
ExecutorService fixed = Executors.newFixedThreadPool(4);
// Internal: ThreadPoolExecutor(4, 4, 0L, TimeUnit.MILLISECONDS, LinkedBlockingQueue)

// Cached thread pool
ExecutorService cached = Executors.newCachedThreadPool();
// Internal: ThreadPoolExecutor(0, Integer.MAX_VALUE, 60L, TimeUnit.SECONDS, SynchronousQueue)

// Single thread executor
ExecutorService single = Executors.newSingleThreadExecutor();
// Internal: ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, LinkedBlockingQueue)

// Scheduled thread pool
ScheduledExecutorService scheduled = Executors.newScheduledThreadPool(2);
```

### Custom ThreadPoolExecutor

```java
public class CustomThreadPool {
    private final ThreadPoolExecutor executor;
    
    public CustomThreadPool() {
        executor = new ThreadPoolExecutor(
            2,                              // corePoolSize
            4,                              // maximumPoolSize
            60L,                            // keepAliveTime
            TimeUnit.SECONDS,               // unit
            new ArrayBlockingQueue<>(100),  // workQueue
            new ThreadFactory() {           // threadFactory
                private final AtomicInteger counter = new AtomicInteger(0);
                @Override
                public Thread newThread(Runnable r) {
                    Thread t = new Thread(r, "CustomPool-" + counter.incrementAndGet());
                    t.setDaemon(false);
                    return t;
                }
            },
            new ThreadPoolExecutor.CallerRunsPolicy() // rejectionHandler
        );
    }
}
```

**Rejection Policies:**
- `AbortPolicy`: Throws RejectedExecutionException (default)
- `CallerRunsPolicy`: Runs task in caller thread
- `DiscardPolicy`: Silently discards task
- `DiscardOldestPolicy`: Discards oldest unhandled task

### Future and CompletableFuture

```java
public class FutureExample {
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    public void basicFuture() throws Exception {
        Future<String> future = executor.submit(() -> {
            Thread.sleep(1000);
            return "Result";
        });
        
        String result = future.get(2, TimeUnit.SECONDS); // Timeout
        System.out.println(result);
    }
    
    public void completableFutureChaining() {
        CompletableFuture<String> future = CompletableFuture
            .supplyAsync(() -> "Hello")
            .thenApply(s -> s + " World")
            .thenApply(String::toUpperCase)
            .thenCompose(s -> CompletableFuture.supplyAsync(() -> s + "!"))
            .exceptionally(throwable -> "Error: " + throwable.getMessage());
        
        future.thenAccept(System.out::println);
    }
    
    public void combiningFutures() {
        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> "World");
        
        CompletableFuture<String> combined = future1.thenCombine(future2, 
            (s1, s2) -> s1 + " " + s2);
        
        combined.thenAccept(System.out::println);
    }
}
```

---

## 6. Advanced Concurrency Utilities - Deep Dive

### CountDownLatch

```java
public class CountDownLatchExample {
    private final CountDownLatch startSignal = new CountDownLatch(1);
    private final CountDownLatch doneSignal = new CountDownLatch(3);
    
    public void example() throws InterruptedException {
        // Start 3 worker threads
        for (int i = 0; i < 3; i++) {
            new Thread(new Worker()).start();
        }
        
        // Let all workers proceed
        startSignal.countDown();
        
        // Wait for all workers to complete
        doneSignal.await();
        System.out.println("All workers completed");
    }
    
    class Worker implements Runnable {
        @Override
        public void run() {
            try {
                startSignal.await(); // Wait for start signal
                doWork();
                doneSignal.countDown(); // Signal completion
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        
        private void doWork() {
            // Simulate work
            try { Thread.sleep(1000); } catch (InterruptedException e) {}
        }
    }
}
```

### CyclicBarrier

```java
public class CyclicBarrierExample {
    private final CyclicBarrier barrier;
    private final int numThreads = 3;
    
    public CyclicBarrierExample() {
        barrier = new CyclicBarrier(numThreads, () -> {
            System.out.println("All threads reached barrier, proceeding...");
        });
    }
    
    public void example() {
        for (int i = 0; i < numThreads; i++) {
            new Thread(new Worker(i)).start();
        }
    }
    
    class Worker implements Runnable {
        private final int id;
        
        Worker(int id) { this.id = id; }
        
        @Override
        public void run() {
            try {
                for (int phase = 0; phase < 3; phase++) {
                    doWork(phase);
                    System.out.println("Thread " + id + " completed phase " + phase);
                    barrier.await(); // Wait for all threads
                }
            } catch (InterruptedException | BrokenBarrierException e) {
                Thread.currentThread().interrupt();
            }
        }
        
        private void doWork(int phase) {
            try { Thread.sleep(1000); } catch (InterruptedException e) {}
        }
    }
}
```

### Semaphore

```java
public class SemaphoreExample {
    private final Semaphore semaphore = new Semaphore(3); // 3 permits
    
    public void accessResource() {
        try {
            semaphore.acquire(); // Acquire permit
            try {
                // Access shared resource
                System.out.println("Accessing resource: " + Thread.currentThread().getName());
                Thread.sleep(2000);
            } finally {
                semaphore.release(); // Always release in finally
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    // Fair semaphore ensures FIFO ordering
    private final Semaphore fairSemaphore = new Semaphore(3, true);
}
```

### Fork-Join Framework

```java
public class ForkJoinExample extends RecursiveTask<Long> {
    private final int[] array;
    private final int start, end;
    private static final int THRESHOLD = 1000;
    
    public ForkJoinExample(int[] array, int start, int end) {
        this.array = array;
        this.start = start;
        this.end = end;
    }
    
    @Override
    protected Long compute() {
        if (end - start <= THRESHOLD) {
            // Base case: compute directly
            long sum = 0;
            for (int i = start; i < end; i++) {
                sum += array[i];
            }
            return sum;
        } else {
            // Divide and conquer
            int mid = (start + end) / 2;
            ForkJoinExample leftTask = new ForkJoinExample(array, start, mid);
            ForkJoinExample rightTask = new ForkJoinExample(array, mid, end);
            
            leftTask.fork(); // Execute asynchronously
            Long rightResult = rightTask.compute(); // Execute in current thread
            Long leftResult = leftTask.join(); // Wait for result
            
            return leftResult + rightResult;
        }
    }
    
    public static void main(String[] args) {
        int[] array = new int[10000];
        // Initialize array...
        
        ForkJoinPool pool = new ForkJoinPool();
        ForkJoinExample task = new ForkJoinExample(array, 0, array.length);
        Long result = pool.invoke(task);
        System.out.println("Sum: " + result);
    }
}
```

---

## 7. Common Concurrency Problems and Solutions

### Deadlock Example and Prevention

```java
public class DeadlockExample {
    private final Object lock1 = new Object();
    private final Object lock2 = new Object();
    
    // Potential deadlock
    public void method1() {
        synchronized(lock1) {
            synchronized(lock2) {
                // Do work
            }
        }
    }
    
    public void method2() {
        synchronized(lock2) { // Different order!
            synchronized(lock1) {
                // Do work
            }
        }
    }
    
    // Solution: Consistent lock ordering
    public void safeMethod1() {
        synchronized(lock1) {
            synchronized(lock2) {
                // Do work
            }
        }
    }
    
    public void safeMethod2() {
        synchronized(lock1) { // Same order
            synchronized(lock2) {
                // Do work
            }
        }
    }
}
```

### Race Condition Example

```java
public class RaceConditionExample {
    private int counter = 0;
    
    // Race condition: read-modify-write is not atomic
    public void unsafeIncrement() {
        counter++; // Actually: temp = counter; temp++; counter = temp;
    }
    
    // Solutions:
    public synchronized void safeIncrement1() {
        counter++;
    }
    
    private final AtomicInteger atomicCounter = new AtomicInteger(0);
    public void safeIncrement2() {
        atomicCounter.incrementAndGet();
    }
}
```

---

## 8. Performance and Monitoring

### Thread Dump Analysis

```java
public class ThreadDumpExample {
    public static void generateThreadDump() {
        ThreadMXBean threadMX = ManagementFactory.getThreadMXBean();
        ThreadInfo[] threadInfos = threadMX.dumpAllThreads(true, true);
        
        for (ThreadInfo threadInfo : threadInfos) {
            System.out.println("Thread: " + threadInfo.getThreadName());
            System.out.println("State: " + threadInfo.getThreadState());
            
            if (threadInfo.getLockName() != null) {
                System.out.println("Waiting on: " + threadInfo.getLockName());
            }
            
            StackTraceElement[] stackTrace = threadInfo.getStackTrace();
            for (StackTraceElement element : stackTrace) {
                System.out.println("\tat " + element);
            }
            System.out.println();
        }
    }
}
```

### Performance Monitoring

```java
public class PerformanceMonitoring {
    private final ThreadPoolExecutor executor;
    
    public PerformanceMonitoring() {
        executor = new ThreadPoolExecutor(
            2, 4, 60L, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(100)
        ) {
            @Override
            protected void beforeExecute(Thread t, Runnable r) {
                super.beforeExecute(t, r);
                System.out.println("Starting task: " + r);
            }
            
            @Override
            protected void afterExecute(Runnable r, Throwable t) {
                super.afterExecute(r, t);
                if (t != null) {
                    System.err.println("Task failed: " + t);
                }
            }
        };
    }
    
    public void printStats() {
        System.out.println("Active threads: " + executor.getActiveCount());
        System.out.println("Completed tasks: " + executor.getCompletedTaskCount());
        System.out.println("Queue size: " + executor.getQueue().size());
    }
}
```

---

## 9. Q&A Preparation - Common Interview Questions

### Q1: What's the difference between synchronized and ReentrantLock?

**Answer:**
- **Flexibility**: ReentrantLock offers tryLock(), timed locking, interruptible locking
- **Fairness**: ReentrantLock can be fair (FIFO), synchronized is unfair
- **Condition Variables**: ReentrantLock supports multiple condition variables
- **Performance**: Similar in modern JVMs
- **Usage**: synchronized is simpler, ReentrantLock when you need advanced features

### Q2: How does ConcurrentHashMap work internally?

**Answer:**
- **Java 7**: Segment-based locking, array of segments each with own lock
- **Java 8+**: Node-based with CAS operations, synchronized only for tree operations
- **Key benefits**: Better scalability, reduced lock contention, optimistic locking

### Q3: Explain the Java Memory Model

**Answer:**
- **Purpose**: Defines when changes made by one thread become visible to others
- **Happens-before**: Relationship that guarantees memory visibility
- **Key rules**: Program order, monitor lock, volatile variable, thread start/join
- **Practical impact**: Determines correctness of concurrent programs

### Q4: When would you use CountDownLatch vs CyclicBarrier?

**Answer:**
- **CountDownLatch**: One-time event, threads wait for signal, cannot be reset
- **CyclicBarrier**: Reusable, all threads wait for each other, can be reset
- **Use CountDownLatch**: Service startup, waiting for initialization
- **Use CyclicBarrier**: Parallel algorithms with phases, iterative computations

### Q5: How do you handle deadlocks?

**Answer:**
- **Prevention**: Consistent lock ordering, timeout-based locking
- **Detection**: Thread dumps, monitoring tools, deadlock detection algorithms
- **Recovery**: Interrupt threads, restart components
- **Avoidance**: Minimize lock scope, use higher-level concurrency utilities

### Q6: What's the difference between submit() and execute()?

**Answer:**
- **execute()**: Executor interface, void return, for Runnable only
- **submit()**: ExecutorService interface, returns Future, accepts Runnable/Callable
- **Exception handling**: submit() captures exceptions in Future, execute() uses UncaughtExceptionHandler

### Q7: Explain volatile keyword

**Answer:**
- **Visibility**: Changes immediately visible to all threads
- **Ordering**: Prevents reordering around volatile operations
- **Not atomic**: Compound operations still need synchronization
- **Use cases**: Flags, status variables, double-checked locking

### Q8: What are the problems with double-checked locking?

**Answer:**
- **Reordering**: Object construction can be reordered
- **Partial construction**: Reference assigned before object fully initialized
- **Solution**: Use volatile keyword for the instance variable
- **Alternative**: Use initialization-on-demand holder pattern

---

## 10. Advanced Topics for Deep Understanding

### Lock-Free Programming

```java
public class LockFreeStack<T> {
    private volatile Node<T> head;
    
    private static class Node<T> {
        final T data;
        volatile Node<T> next;
        
        Node(T data) { this.data = data; }
    }
    
    public void push(T item) {
        Node<T> newNode = new Node<>(item);
        Node<T> currentHead;
        do {
            currentHead = head;
            newNode.next = currentHead;
        } while (!compareAndSetHead(currentHead, newNode));
    }
    
    public T pop() {
        Node<T> currentHead;
        Node<T> newHead;
        do {
            currentHead = head;
            if (currentHead == null) return null;
            newHead = currentHead.next;
        } while (!compareAndSetHead(currentHead, newHead));
        
        return currentHead.data;
    }
    
    private boolean compareAndSetHead(Node<T> expected, Node<T> update) {
        // Atomic compare-and-swap operation
        // In real implementation, would use Unsafe or VarHandle
        return true; // Simplified
    }
}
```

### Virtual Threads (Project Loom - Java 19+)

```java
public class VirtualThreadExample {
    public void traditionalThreads() throws InterruptedException {
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            Thread t = new Thread(() -> {
                try { Thread.sleep(1000); } catch (InterruptedException e) {}
            });
            threads.add(t);
            t.start();
        }
        
        for (Thread t : threads) {
            t.join();
        }
    }
    
    public void virtualThreads() throws InterruptedException {
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < 1_000_000; i++) { // Much higher number!
                Future<?> future = executor.submit(() -> {
                    try { Thread.sleep(1000); } catch (InterruptedException e) {}
                });
                futures.add(future);
            }
            
            for (Future<?> future : futures) {
                future.get();
            }
        }
    }
}
```