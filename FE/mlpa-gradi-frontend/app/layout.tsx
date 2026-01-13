import type { Metadata } from "next";
import "./globals.css";


export const metadata: Metadata = {
  title: "Gradi",
  description: "Auto Grading Gradi",
  icons: {
    icon: "/dku_emblem_icon.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link href="https://hangeul.pstatic.net/hangeul_static/css/nanum-square.css" rel="stylesheet" />
      </head>
      <body
        suppressHydrationWarning
        className="antialiased"
      >
        {children}
      </body>
    </html>
  );
}
