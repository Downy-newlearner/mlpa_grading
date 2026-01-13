package com.dankook.mlpa_gradi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableJpaAuditing
@SpringBootApplication
@EnableScheduling
public class MlpaGradiApplication {

    public static void main(String[] args) {
        SpringApplication.run(MlpaGradiApplication.class, args);
    }
}
