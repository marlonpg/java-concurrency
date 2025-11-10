# Java Concurrency Deep Study Guide V2 - Extended Learning

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

---

## 5. Extended Learning Topics

### Thread Pool Internals and Tuning

**ThreadPoolExecutor Parameters Deep Dive:**
```java
public class ThreadPoolTuning {
    // Core parameters explanation
    ThreadPoolExecutor executor = new ThreadPoolExecutor(
        2,                                    // corePoolSize: minimum threads kept alive
        10,                                   // maximumPoolSize: maximum threads allowed
        60L,                                  // keepAliveTime: idle thread timeout
        TimeUnit.SECONDS,                     // time unit for keepAliveTime
        new LinkedBlockingQueue<>(1000),      // workQueue: task queue
        Executors.defaultThreadFactory(),     // threadFactory: creates new threads
        new ThreadPoolExecutor.AbortPolicy()  // rejectionHandler: handles rejected tasks
    );
    
    // Monitoring thread pool health
    public void monitorThreadPool() {
        System.out.println("Active threads: " + executor.getActiveCount());
        System.out.println("Pool size: " + executor.getPoolSize());
        System.out.println("Core pool size: " + executor.getCorePoolSize());
        System.out.println("Maximum pool size: " + executor.getMaximumPoolSize());
        System.out.println("Task count: " + executor.getTaskCount());
        System.out.println("Completed tasks: " + executor.getCompletedTaskCount());
        System.out.println("Queue size: " + executor.getQueue().size());
    }
}
```

**Thread Pool Sizing Guidelines:**
- **CPU-intensive tasks**: Number of cores + 1
- **I/O-intensive tasks**: Number of cores * (1 + Wait time / Service time)
- **Mixed workloads**: Profile and test different configurations

### Advanced Synchronization Patterns

**Producer-Consumer with Multiple Conditions:**
```java
public class AdvancedProducerConsumer<T> {
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();
    private final Queue<T> queue = new LinkedList<>();
    private final int capacity;
    
    public AdvancedProducerConsumer(int capacity) {
        this.capacity = capacity;
    }
    
    public void produce(T item) throws InterruptedException {
        lock.lock();
        try {
            while (queue.size() == capacity) {
                notFull.await(); // Wait for space
            }
            queue.offer(item);
            notEmpty.signal(); // Signal consumers
        } finally {
            lock.unlock();
        }
    }
    
    public T consume() throws InterruptedException {
        lock.lock();
        try {
            while (queue.isEmpty()) {
                notEmpty.await(); // Wait for items
            }
            T item = queue.poll();
            notFull.signal(); // Signal producers
            return item;
        } finally {
            lock.unlock();
        }
    }
}
```

### Lock-Free Data Structures

**Lock-Free Queue Implementation:**
```java
public class LockFreeQueue<T> {
    private final AtomicReference<Node<T>> head;
    private final AtomicReference<Node<T>> tail;
    
    private static class Node<T> {
        volatile T data;
        volatile Node<T> next;
        
        Node(T data) {
            this.data = data;
        }
    }
    
    public LockFreeQueue() {
        Node<T> dummy = new Node<>(null);
        head = new AtomicReference<>(dummy);
        tail = new AtomicReference<>(dummy);
    }
    
    public void enqueue(T item) {
        Node<T> newNode = new Node<>(item);
        while (true) {
            Node<T> last = tail.get();
            Node<T> next = last.next;
            
            if (last == tail.get()) { // Consistency check
                if (next == null) {
                    // Try to link new node
                    if (compareAndSetNext(last, null, newNode)) {
                        tail.compareAndSet(last, newNode); // Move tail
                        break;
                    }
                } else {
                    // Help move tail
                    tail.compareAndSet(last, next);
                }
            }
        }
    }
    
    public T dequeue() {
        while (true) {
            Node<T> first = head.get();
            Node<T> last = tail.get();
            Node<T> next = first.next;
            
            if (first == head.get()) { // Consistency check
                if (first == last) {
                    if (next == null) {
                        return null; // Queue is empty
                    }
                    tail.compareAndSet(last, next); // Help move tail
                } else {
                    T data = next.data;
                    if (head.compareAndSet(first, next)) {
                        return data;
                    }
                }
            }
        }
    }
    
    private boolean compareAndSetNext(Node<T> node, Node<T> expected, Node<T> update) {
        // Simplified - would use VarHandle or Unsafe in real implementation
        return true;
    }
}
```

### Memory Barriers and Ordering

**Memory Ordering Modes (VarHandle):**
```java
public class MemoryOrderingExample {
    private static final VarHandle VALUE_HANDLE;
    private volatile int value;
    
    static {
        try {
            VALUE_HANDLE = MethodHandles.lookup()
                .findVarHandle(MemoryOrderingExample.class, "value", int.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    // Different memory ordering modes
    public void demonstrateOrdering() {
        // Plain access - no ordering guarantees
        VALUE_HANDLE.set(this, 42);
        int plain = (int) VALUE_HANDLE.get(this);
        
        // Opaque access - atomicity but no ordering
        VALUE_HANDLE.setOpaque(this, 42);
        int opaque = (int) VALUE_HANDLE.getOpaque(this);
        
        // Acquire/Release ordering
        VALUE_HANDLE.setRelease(this, 42); // Release semantics
        int acquire = (int) VALUE_HANDLE.getAcquire(this); // Acquire semantics
        
        // Volatile access - full ordering
        VALUE_HANDLE.setVolatile(this, 42);
        int vol = (int) VALUE_HANDLE.getVolatile(this);
    }
}
```

### Performance Optimization Techniques

**False Sharing Prevention:**
```java
// Problem: False sharing
class CounterWithFalseSharing {
    private volatile long counter1 = 0;
    private volatile long counter2 = 0; // Likely in same cache line
}

// Solution: Padding
class CounterWithPadding {
    private volatile long counter1 = 0;
    private long p1, p2, p3, p4, p5, p6, p7; // Padding
    private volatile long counter2 = 0;
}

// Java 8+ Solution: @Contended annotation
@jdk.internal.vm.annotation.Contended
class CounterWithContended {
    private volatile long counter1 = 0;
    private volatile long counter2 = 0;
}
```

**Thread-Local Optimization:**
```java
public class ThreadLocalOptimization {
    // Expensive to create, so cache per thread
    private static final ThreadLocal<SimpleDateFormat> DATE_FORMAT = 
        ThreadLocal.withInitial(() -> new SimpleDateFormat("yyyy-MM-dd"));
    
    // Better: Use thread-local random
    private static final ThreadLocal<Random> THREAD_LOCAL_RANDOM = 
        ThreadLocal.withInitial(Random::new);
    
    // Java 7+: Even better, use ThreadLocalRandom
    public int getRandomNumber() {
        return ThreadLocalRandom.current().nextInt(100);
    }
}
```

### Reactive Programming Patterns

**CompletableFuture Advanced Patterns:**
```java
public class ReactivePatterns {
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    // Timeout pattern
    public CompletableFuture<String> withTimeout(Supplier<String> task, Duration timeout) {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(task, executor);
        
        CompletableFuture<String> timeoutFuture = new CompletableFuture<>();
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        scheduler.schedule(() -> {
            timeoutFuture.completeExceptionally(new TimeoutException("Task timed out"));
        }, timeout.toMillis(), TimeUnit.MILLISECONDS);
        
        return future.applyToEither(timeoutFuture, Function.identity());
    }
    
    // Retry pattern
    public CompletableFuture<String> withRetry(Supplier<String> task, int maxRetries) {
        return CompletableFuture.supplyAsync(task, executor)
            .handle((result, throwable) -> {
                if (throwable != null && maxRetries > 0) {
                    return withRetry(task, maxRetries - 1).join();
                }
                if (throwable != null) {
                    throw new RuntimeException(throwable);
                }
                return result;
            });
    }
    
    // Circuit breaker pattern
    public class CircuitBreaker {
        private final int failureThreshold;
        private final Duration timeout;
        private int failureCount = 0;
        private long lastFailureTime = 0;
        private volatile State state = State.CLOSED;
        
        enum State { CLOSED, OPEN, HALF_OPEN }
        
        public CircuitBreaker(int failureThreshold, Duration timeout) {
            this.failureThreshold = failureThreshold;
            this.timeout = timeout;
        }
        
        public <T> CompletableFuture<T> execute(Supplier<CompletableFuture<T>> task) {
            if (state == State.OPEN) {
                if (System.currentTimeMillis() - lastFailureTime > timeout.toMillis()) {
                    state = State.HALF_OPEN;
                } else {
                    return CompletableFuture.failedFuture(
                        new RuntimeException("Circuit breaker is OPEN"));
                }
            }
            
            return task.get()
                .whenComplete((result, throwable) -> {
                    if (throwable != null) {
                        onFailure();
                    } else {
                        onSuccess();
                    }
                });
        }
        
        private void onFailure() {
            failureCount++;
            lastFailureTime = System.currentTimeMillis();
            if (failureCount >= failureThreshold) {
                state = State.OPEN;
            }
        }
        
        private void onSuccess() {
            failureCount = 0;
            state = State.CLOSED;
        }
    }
}
```

---

## 6. Advanced Debugging and Monitoring

### Thread Dump Analysis Tools

**Programmatic Thread Dump Generation:**
```java
public class ThreadDumpAnalyzer {
    public static void generateDetailedThreadDump() {
        ThreadMXBean threadMX = ManagementFactory.getThreadMXBean();
        
        // Enable CPU time measurement if supported
        if (threadMX.isThreadCpuTimeSupported()) {
            threadMX.setThreadCpuTimeEnabled(true);
        }
        
        ThreadInfo[] threadInfos = threadMX.dumpAllThreads(true, true);
        
        for (ThreadInfo threadInfo : threadInfos) {
            System.out.println("=== Thread: " + threadInfo.getThreadName() + " ===");
            System.out.println("ID: " + threadInfo.getThreadId());
            System.out.println("State: " + threadInfo.getThreadState());
            
            if (threadInfo.getLockName() != null) {
                System.out.println("Blocked on: " + threadInfo.getLockName());
                System.out.println("Lock owner: " + threadInfo.getLockOwnerName());
            }
            
            // CPU time information
            long cpuTime = threadMX.getThreadCpuTime(threadInfo.getThreadId());
            if (cpuTime != -1) {
                System.out.println("CPU time: " + cpuTime / 1_000_000 + " ms");
            }
            
            // Stack trace
            StackTraceElement[] stackTrace = threadInfo.getStackTrace();
            for (StackTraceElement element : stackTrace) {
                System.out.println("\tat " + element);
            }
            
            // Locked monitors and synchronizers
            MonitorInfo[] monitors = threadInfo.getLockedMonitors();
            for (MonitorInfo monitor : monitors) {
                System.out.println("Locked monitor: " + monitor);
            }
            
            LockInfo[] synchronizers = threadInfo.getLockedSynchronizers();
            for (LockInfo sync : synchronizers) {
                System.out.println("Locked synchronizer: " + sync);
            }
            
            System.out.println();
        }
    }
    
    // Deadlock detection
    public static void detectDeadlocks() {
        ThreadMXBean threadMX = ManagementFactory.getThreadMXBean();
        long[] deadlockedThreads = threadMX.findDeadlockedThreads();
        
        if (deadlockedThreads != null) {
            System.out.println("DEADLOCK DETECTED!");
            ThreadInfo[] threadInfos = threadMX.getThreadInfo(deadlockedThreads);
            for (ThreadInfo threadInfo : threadInfos) {
                System.out.println("Deadlocked thread: " + threadInfo.getThreadName());
                System.out.println("Waiting for: " + threadInfo.getLockName());
                System.out.println("Owned by: " + threadInfo.getLockOwnerName());
            }
        }
    }
}
```

### JVM Flags for Concurrency Debugging

**Useful JVM flags:**
```bash
# Enable detailed GC logging
-XX:+PrintGC -XX:+PrintGCDetails -XX:+PrintGCTimeStamps

# Thread stack size
-Xss1m

# Detect JVM deadlocks
-XX:+PrintConcurrentLocks

# Enable JFR (Java Flight Recorder)
-XX:+FlightRecorder -XX:StartFlightRecording=duration=60s,filename=app.jfr

# Enable detailed thread information
-XX:+PrintGCApplicationStoppedTime
-XX:+PrintStringDeduplicationStatistics
```

---

## 7. Testing Concurrent Code

### JCStress Framework

**Stress Testing Example:**
```java
@JCStressTest
@Outcome(id = "0, 0", expect = Expect.ACCEPTABLE, desc = "Both threads see initial value")
@Outcome(id = "1, 0", expect = Expect.ACCEPTABLE, desc = "Thread 1 updated first")
@Outcome(id = "0, 1", expect = Expect.ACCEPTABLE, desc = "Thread 2 updated first")
@Outcome(id = "1, 1", expect = Expect.FORBIDDEN, desc = "Both threads see each other's update")
@State
public class RaceConditionTest {
    int x = 0;
    int y = 0;
    
    @Actor
    public void actor1(II_Result r) {
        x = 1;
        r.r1 = y;
    }
    
    @Actor
    public void actor2(II_Result r) {
        y = 1;
        r.r2 = x;
    }
}
```

### Property-Based Testing

**QuickCheck-style testing:**
```java
public class ConcurrentPropertyTest {
    @Test
    public void testAtomicCounterProperties() {
        AtomicInteger counter = new AtomicInteger(0);
        int numThreads = 10;
        int incrementsPerThread = 1000;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch latch = new CountDownLatch(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < incrementsPerThread; j++) {
                        counter.incrementAndGet();
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        try {
            latch.await();
            assertEquals(numThreads * incrementsPerThread, counter.get());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
        } finally {
            executor.shutdown();
        }
    }
}
```

---

## References

### Official Documentation
- [Oracle Java Concurrency Tutorial](https://docs.oracle.com/javase/tutorial/essential/concurrency/)
- [java.util.concurrent Package](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/package-summary.html)
- [Java Memory Model Specification](https://docs.oracle.com/javase/specs/jls/se21/html/jls-17.html)

### JSR Specifications
- [JSR-166: Concurrency Utilities](https://jcp.org/en/jsr/detail?id=166)
- [JSR-133: Java Memory Model](https://jcp.org/en/jsr/detail?id=133)

### Essential Books
- **"Java Concurrency in Practice"** by Brian Goetz
- **"Effective Java"** by Joshua Bloch (Items 78-84)

### Project Loom
- [JEP 444: Virtual Threads](https://openjdk.org/jeps/444)
- [JEP 453: Structured Concurrency](https://openjdk.org/jeps/453)

### Advanced Features
- [JEP 193: Variable Handles](https://openjdk.org/jeps/193)
- [JEP 429: Scoped Values](https://openjdk.org/jeps/429)
- [Doug Lea's Papers](http://gee.cs.oswego.edu/dl/papers/)

### Performance Tools
- [JMH Microbenchmark Harness](https://openjdk.org/projects/code-tools/jmh/)
- [JCStress Concurrency Testing](https://openjdk.org/projects/code-tools/jcstress/)