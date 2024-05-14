package com.example.hpcc;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.web.servlet.error.ErrorMvcAutoConfiguration;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication(exclude = {ErrorMvcAutoConfiguration.class})
@ComponentScan(basePackages = "com.example.controller")
public class HpccApplication {

	public static void main(String[] args) {

		SpringApplication.run(HpccApplication.class, args);
	}

}
