package com.dankook.mlpa_gradi.repository.memory;

import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Repository
public class InMemoryReportRepository {

    // examCode -> List of Image URLs
    private final Map<String, List<String>> unknownImagesCache = new ConcurrentHashMap<>();

    public void saveUnknownImages(String examCode, List<String> urls) {
        unknownImagesCache.computeIfAbsent(examCode, k -> new ArrayList<>()).addAll(urls);
    }

    public List<String> getUnknownImages(String examCode) {
        return unknownImagesCache.getOrDefault(examCode, new ArrayList<>());
    }

    public void clear(String examCode) {
        unknownImagesCache.remove(examCode);
    }
}
