package com.gamba.software.concurrency;

import java.util.concurrent.atomic.AtomicInteger;

public class AtomicExample {
    private final AtomicInteger counter = new AtomicInteger(0);

    public int increment() {
        return counter.incrementAndGet(); // Atomic operation
    }

    public boolean compareAndSet(int expected, int update) {
        return counter.compareAndSet(expected, update);
    }

    public int addTen() {
        return counter.updateAndGet(current -> current + 10);
    }

    public static void main(String[] args) throws InterruptedException {
        AtomicExample example = new AtomicExample();

        // Create 1000 threads that increment counter
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(()-> {
                example.increment();
                try {
                    Thread.sleep(21000);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            });
            threads[i].start();
        }

        // Wait for all threads
        for (Thread t : threads) {
            System.out.println(t.getName());
            t.join();
        }

        System.out.println("Final count: " + example.counter.get());
        // Will always be 1000 - no race condition!
    }
}
