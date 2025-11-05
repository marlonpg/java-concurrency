package com.gamba.software.concurrency;

public class ScopedValueExample {
    private static final ScopedValue<String> USER_ID = ScopedValue.newInstance();
    private static final ScopedValue<String> REQUEST_ID = ScopedValue.newInstance();
    
    public void handleRequest(String userId, String requestId) {
        ScopedValue.where(USER_ID, userId)
                  .where(REQUEST_ID, requestId)
                  .run(() -> {
                      processRequest();
                      // USER_ID and REQUEST_ID are available in all called methods
                  });
        // Values automatically cleaned up here
    }
    
    private void processRequest() {
        String currentUser = USER_ID.get(); // Available without passing parameters
        String currentRequest = REQUEST_ID.get();
        
        // Call other methods that can also access these values
        validateUser();
        logRequest();
    }
    
    private void validateUser() {
        String userId = USER_ID.get(); // Still available
        // Validation logic
        System.out.println("Validating user: " + userId);
    }
    
    private void logRequest() {
        String userId = USER_ID.get();
        String requestId = REQUEST_ID.get();
        System.out.println("Processing request " + requestId + " for user " + userId);
    }
}