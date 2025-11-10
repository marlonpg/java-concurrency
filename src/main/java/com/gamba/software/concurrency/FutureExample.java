package com.gamba.software.concurrency;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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