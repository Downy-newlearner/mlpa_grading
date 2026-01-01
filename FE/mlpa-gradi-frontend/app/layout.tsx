import type { Metadata } from "next";
import "./globals.css";
import AutoRefresh from "./components/AutoRefresh";

export const metadata: Metadata = {
  title: "Gradi(MLPA)",
  description: "Auto Grading Gradi",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link href="https://hangeul.pstatic.net/hangeul_static/css/nanum-square.css" rel="stylesheet" />
      </head>
      <body
        className="antialiased"
      >
        <AutoRefresh />
        {children}
      </body>
    </html>
  );
}
