package com.gamba.software.concurrency;


public class ExampleSynchronized {

    public synchronized void doWork() {
        System.out.println(Thread.currentThread().getName() + " entered doWork");
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        System.out.println(Thread.currentThread().getName() + " leaving doWork");
    }

    public static void main(String[] args) throws InterruptedException {
        ExampleSynchronized ex = new ExampleSynchronized();

        Thread t1 = new Thread(ex::doWork, "T1");
        Thread t2 = new Thread(ex::doWork, "T2");

        t1.start();
        Thread.sleep(100); // small delay to ensure T1 acquires the lock first
        t2.start();

        // Log the states while threads are running
        for (int i = 0; i < 5; i++) {
            System.out.printf("T1 state: %s | T2 state: %s%n",
                    t1.getState(), t2.getState());
            Thread.sleep(500);
        }

        t1.join();
        t2.join();

        System.out.println("Final states:");
        System.out.printf("T1: %s | T2: %s%n", t1.getState(), t2.getState());
    }
}