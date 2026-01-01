package com.dankook.mlpa_gradi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;

@EnableJpaAuditing
@SpringBootApplication
public class MlpaGradiApplication {

    public static void main(String[] args) {
        SpringApplication.run(MlpaGradiApplication.class, args);
    }
}
