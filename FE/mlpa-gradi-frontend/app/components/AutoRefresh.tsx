"use client";

import { useEffect } from "react";

export default function AutoRefresh() {
    useEffect(() => {
        const INTERVAL = 5 * 60 * 1000; // 5분

        const timer = setInterval(() => {
            console.log("Auto-refreshing page...");
            window.location.reload();
        }, INTERVAL);

        return () => clearInterval(timer);
    }, []);

    return null;
}
