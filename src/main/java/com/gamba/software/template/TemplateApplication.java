package com.gamba.software.template;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring Boot application entry point.
 * Provides the main class for running the concurrency examples as a Spring Boot application.
 */
@SpringBootApplication // Enables auto-configuration, component scanning, and configuration
public class TemplateApplication {

	/**
	 * Main method to start the Spring Boot application.
	 * Initializes the Spring context and starts the embedded server.
	 */
	public static void main(String[] args) {
		SpringApplication.run(TemplateApplication.class, args); // Bootstrap Spring Boot
	}

}
