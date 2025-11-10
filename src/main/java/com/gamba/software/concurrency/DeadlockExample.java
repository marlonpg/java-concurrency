package com.gamba.software.concurrency;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

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

        // Give threads time to deadlock
        Thread.sleep(2000);

        // Detect deadlock
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