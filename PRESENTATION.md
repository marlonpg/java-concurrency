# Java Concurrency Deep Dive

## 1. What Concurrency Means

**Concurrency** is the ability to execute multiple tasks seemingly at the same time to improve performance, responsiveness, and resource utilization.
- **Challenge**: Managing concurrent access to shared resources is complex. Without careful synchronization, you risk data inconsistencies, race conditions, and deadlocks.

https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/util/concurrent/package-summary.html

---

## 2. Threads

Threads are the foundation of concurrency in Java. Every Java program starts with at least one thread (main thread).

**Thread Creation Methods:**

**Method 1: Extending Thread class**
```java
class MyThread extends Thread {
    @Override
    public void run() {
        System.out.println("Running in: " + getName());
    }
}

MyThread thread = new MyThread();
thread.start();
```

**Method 2: Implementing Runnable**
```java
class MyTask implements Runnable {
    @Override
    public void run() {
        System.out.println("Task running in: " + Thread.currentThread().getName());
    }
}

Thread thread = new Thread(new MyTask());
thread.start();
```

**Method 3: Lambda Expression**
```java
Thread thread = new Thread(() -> {
    System.out.println("Lambda thread: " + Thread.currentThread().getName());
});
thread.start();
```

**Important Thread Methods:**
- `start()` vs `run()`: Always use `start()` to create a new thread
- `join()`: Wait for thread completion
- `interrupt()`: Cooperative cancellation
- `sleep()`: Pause execution

---

## 3. Problems with Manual Thread Management

Manual thread creation and management leads to several critical issues:

**Problem 1: Race Conditions**
```java
class Counter {
    private int count = 0;

    public void increment() {
        count++;
    }

    public int getCount() {
        return count;
    }
}
public class RaceConditionExample {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        });

        t1.start();
        t2.start();

        t1.join();
        t2.join();

        System.out.println("Final count: " + counter.getCount());
    }
}
```

**Problem 2: Resource Management**
- Thread creation is expensive
- Unlimited thread creation can exhaust system resources
- Difficult to control thread lifecycle

**Problem 3: Deadlocks**
```java
public class DeadlockExample {
    private final Object lock1 = new Object();
    private final Object lock2 = new Object();

    public void method1() {
        synchronized (lock1) {
            System.out.println(Thread.currentThread().getName() + " acquired lock1");
            try { Thread.sleep(500); } catch (InterruptedException e) {}
            synchronized (lock2) {
                System.out.println("Method 1 executed");
            }
        }
    }

    public void method2() {
        synchronized (lock2) {
            System.out.println(Thread.currentThread().getName() + " acquired lock2");
            try { Thread.sleep(500); } catch (InterruptedException e) {}
            synchronized (lock1) {
                System.out.println("Method 2 executed");
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        DeadlockExample example = new DeadlockExample();

        Thread t1 = new Thread(example::method1, "Thread-1");
        Thread t2 = new Thread(example::method2, "Thread-2");

        t1.start();
        t2.start();

        //give threads time to deadlock
        Thread.sleep(2000);

        // find and show deadlock threads
        var bean = java.lang.management.ManagementFactory.getThreadMXBean();
        long[] deadlocked = bean.findDeadlockedThreads();
        if (deadlocked != null) {
            System.out.println("Deadlock detected between threads:");
            for (long id : deadlocked) {
                var info = bean.getThreadInfo(id);
                System.out.println(" - " + info.getThreadName());
            }
        } else {
            System.out.println("No deadlock detected.");
        }
    }
}
// Output:
//Thread-1 acquired lock1
//Thread-2 acquired lock2
//Deadlock detected between threads:
// - Thread-1
// - Thread-2
```

---

## 4. Synchronization and Locks

Synchronization ensures thread-safe access to shared resources.

**The `synchronized` Keyword:**
```java
public class SynchronizedExample {
    private volatile int counter = 0;
    
    // Method-level synchronization
    public synchronized void increment() {
        counter++; // Now thread-safe
    }
    
    // Block-level synchronization
    public void decrement() {
        synchronized(this) {
            counter--;
        }
    }
    
    public int getCounter() {
        return counter;
    }
}
```

---

## 5. Executor Framework

Introduced in Java 5, the Executor Framework provides a higher-level API for managing threads through thread pools.

**Common Thread Pool Types:**
```java
public class ExecutorExamples {
    public static void main(String[] args) {
        // Fixed thread pool - reuses fixed number of threads
        ExecutorService fixedPool = Executors.newFixedThreadPool(4);
        
        // Cached thread pool - creates threads as needed
        //ExecutorService cachedPool = Executors.newCachedThreadPool();
        
        // Single thread executor - guarantees sequential execution
        //ExecutorService singleThread = Executors.newSingleThreadExecutor();
        
        // Submit tasks
        fixedPool.submit(() -> {
            System.out.println("Task running in: " + Thread.currentThread().getName());
        });
        
        fixedPool.shutdown();

        // ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        // Runnable task = () -> System.out.println("Running at " + System.currentTimeMillis());

        // // start after 2s, repeat every 5s (fixed rate)
        // scheduler.scheduleAtFixedRate(task, 2, 5, TimeUnit.SECONDS);
    }
}
```

**Benefits:**
- Thread reuse reduces creation overhead
- Better resource management
- Built-in task queuing

---

## 6. Atomic Variables

Lock-free thread-safe operations using Compare-and-Swap (CAS) hardware primitives.

**AtomicInteger - Lock-Free Counter:**
```java
import java.util.concurrent.atomic.AtomicInteger;

public class AtomicExample {
    private final AtomicInteger counter = new AtomicInteger(0);
    
    public int increment() {
        return counter.incrementAndGet();
    }
    
    public boolean compareAndSet(int expected, int update) {
        return counter.compareAndSet(expected, update);
    }
    
    public int addTen() {
        return counter.updateAndGet(current -> current + 10);
    }
    
    public static void main(String[] args) throws InterruptedException {
        AtomicExample example = new AtomicExample();
        
        // creating 1000 threads that increment counter
        Thread[] threads = new Thread[1000];
        for (int i = 0; i < 1000; i++) {
            threads[i] = new Thread(example::increment);
            threads[i].start();
        }
        
        // wait for all threads
        for (Thread t : threads) {
            t.join();
        }
        
        System.out.println("Final count: " + example.counter.get());
        // the result will always be 1000, so no race condition
    }
}
```

---

## 7. CompletableFuture

Modern, non-blocking API for composing asynchronous computations (Java 8+).

**Basic CompletableFuture Usage:**
```java
public class CompletableFutureChaining {
    public static void main(String[] args) throws InterruptedException {
        CompletableFuture<String> result = CompletableFuture
                .supplyAsync(() -> {
                    try { Thread.sleep(100); } catch (Exception e) {}
                    return "Hello";
                })
                .thenApply(s -> s + " World")
                .thenApply(String::toUpperCase)
                .thenCompose(s -> CompletableFuture.supplyAsync(() -> s + "!"))
                .exceptionally(throwable -> "Error: " + throwable.getMessage());

        result.thenAccept(System.out::println);

        System.out.println("Main thread is free");
        Thread.sleep(200);
    }
}
```

**Combining Multiple Futures:**
```java
public class CompletableFutureCombining {
    public static void main(String[] args) {
        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> "World");
        
        // Combine two futures
        CompletableFuture<String> combined = future1.thenCombine(future2, 
            (s1, s2) -> s1 + " " + s2);
        
        combined.thenAccept(System.out::println);
        
        // Wait for all futures
        CompletableFuture<Void> allOf = CompletableFuture.allOf(future1, future2);
        allOf.thenRun(() -> System.out.println("All completed"));
    }
}
```
---

## 8. References
- [Oracle Java Concurrency Tutorial](https://docs.oracle.com/javase/tutorial/essential/concurrency/)
- [Baeldung - Java Concurrency](https://www.baeldung.com/java-concurrency)
- [java.util.concurrent Package](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/package-summary.html)
- [Java Memory Model Specification](https://docs.oracle.com/javase/specs/jls/se21/html/jls-17.html)
- [JEP 444: Virtual Threads](https://openjdk.org/jeps/444)
- [jenkov.com](https://jenkov.com/tutorials/java-concurrency/index.html)
































## EXTRA (if I have time)
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


---

---

## 6. Future and Callable

`Future` and `Callable` allow tasks to return results and handle exceptions asynchronously.

**Callable vs Runnable:**
```java
public class FutureExample {
    public static void main(String[] args) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        
        // Callable returns a value and can throw exceptions
        Callable<String> task = () -> {
            Thread.sleep(1000);
            return "Task completed at " + System.currentTimeMillis();
        };
        
        // Submit Callable and get Future
        Future<String> future = executor.submit(task);
        
        // Do other work while task executes
        System.out.println("Task submitted, doing other work...");
        
        // Get result (blocks until complete)
        String result = future.get();
        System.out.println("Result: " + result);
        
        // Check Future status
        System.out.println("Is done: " + future.isDone());
        System.out.println("Is cancelled: " + future.isCancelled());
        
        executor.shutdown();
    }
}
```

**Future with Timeout:**
```java
try {
    String result = future.get(2, TimeUnit.SECONDS);
} catch (TimeoutException e) {
    System.out.println("Task timed out");
    future.cancel(true); // Interrupt if running
}
```

---

## 8. Concurrent Collections

Thread-safe collections designed for concurrent access without external synchronization.

**ConcurrentHashMap - High-Performance Thread-Safe Map:**
```java
import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentCollectionExample {
    public static void main(String[] args) {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        
        // Thread-safe operations
        map.put("key1", 1);
        map.putIfAbsent("key2", 2);
        
        // Atomic operations
        map.compute("key1", (key, value) -> value + 1);
        map.merge("key3", 1, Integer::sum);
        
        System.out.println(map);
    }
}
```

