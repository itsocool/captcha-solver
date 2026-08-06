package kr.co.hyperinfo.captchaSolver;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class CaptchaSolverApplication {

	public static void main(String[] args) {
		SpringApplication.run(CaptchaSolverApplication.class, args);
	}

}
