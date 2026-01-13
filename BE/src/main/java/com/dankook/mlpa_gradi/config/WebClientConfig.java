package com.dankook.mlpa_gradi.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient aiWebClient(@Value("${ai.server.url}") String aiServerUrl) {
        return WebClient.builder()
                .baseUrl(aiServerUrl)
                .build();
    }
}
