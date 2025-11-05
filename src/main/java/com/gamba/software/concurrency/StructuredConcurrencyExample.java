package com.gamba.software.concurrency;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.StructuredTaskScope;
import java.util.function.Supplier;

public class StructuredConcurrencyExample {
    public String handleRequest(String requestId) throws InterruptedException, ExecutionException {
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            // Fork multiple related tasks
            Supplier<String> task1 = scope.fork(() -> processStep1(requestId));
            Supplier<String> task2 = scope.fork(() -> processStep2(requestId));
            Supplier<String> task3 = scope.fork(() -> processStep3(requestId));
            
            scope.join();           // Wait for all to complete
            scope.throwIfFailed();  // Propagate any failures
            
            // All tasks succeeded
            return combineResults(
                task1.get(),
                task2.get(), 
                task3.get()
            );
        } // Automatic cleanup of any remaining tasks
    }
    
    private String processStep1(String requestId) {
        try { Thread.sleep(100); } catch (InterruptedException e) {}
        return "Step1-" + requestId;
    }
    
    private String processStep2(String requestId) {
        try { Thread.sleep(150); } catch (InterruptedException e) {}
        return "Step2-" + requestId;
    }
    
    private String processStep3(String requestId) {
        try { Thread.sleep(200); } catch (InterruptedException e) {}
        return "Step3-" + requestId;
    }
    
    private String combineResults(String s1, String s2, String s3) {
        return String.format("Combined: %s, %s, %s", s1, s2, s3);
    }
}