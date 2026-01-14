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
        try {
            System.out.println("🔍 Current Directory: " + System.getProperty("user.dir"));
            System.out.println("🔍 Loading .env file...");
            io.github.cdimascio.dotenv.Dotenv dotenv = io.github.cdimascio.dotenv.Dotenv.configure()
                    .directory("./")
                    .ignoreIfMissing()
                    .load();
            dotenv.entries().forEach(entry -> {
                String key = entry.getKey();
                String value = entry.getValue();
                System.out.println("✅ Setting Property: " + key + " = "
                        + (key.contains("PASSWORD") || key.contains("SECRET") ? "********" : value));
                System.setProperty(key, value);
            });
        } catch (Exception e) {
            System.err.println("❌ Error loading .env file: " + e.getMessage());
            e.printStackTrace();
        }
        SpringApplication.run(MlpaGradiApplication.class, args);
    }
}
