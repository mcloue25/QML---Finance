"use client";

import React, { useMemo, useState } from "react";

export function DashboardNavbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  const links = useMemo(
    () => [
        { href: "#run", label: "Run analysis" },
        { href: "#portfolio", label: "Portfolio" },
        { href: "#trades", label: "Trades" },
        { href: "#models", label: "Model comparison" },
    ],
    []
  );

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/70 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
        {/* Brand / title */}
        <a
          className="text-sm font-semibold tracking-tight hover:opacity-90"
          href="#top"
        >
          Backtest Dashboard
        </a>

        <div className="relative flex items-center">
          {/* Mobile toggle */}
          <button
            className="md:hidden inline-flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-muted/30 hover:bg-muted/50 transition"
            onClick={() => setMenuOpen((v) => !v)}
            aria-label="Toggle menu"
            aria-expanded={menuOpen}
          >
            {/* Simple icon (no image dependency) */}
            <svg
              viewBox="0 0 24 24"
              className="h-5 w-5 text-muted-foreground"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              {menuOpen ? (
                <>
                  <path d="M18 6 6 18" />
                  <path d="M6 6l12 12" />
                </>
              ) : (
                <>
                  <path d="M4 6h16" />
                  <path d="M4 12h16" />
                  <path d="M4 18h16" />
                </>
              )}
            </svg>
          </button>

          {/* Desktop links */}
          <ul className="ml-8 hidden list-none items-center gap-7 md:flex">
            {links.map((item) => (
              <li key={item.href}>
                <a
                  href={item.href}
                  className="relative text-sm text-muted-foreground transition hover:text-foreground"
                >
                  <span className="after:absolute after:-bottom-1 after:left-0 after:h-[2px] after:w-0 after:bg-foreground/70 after:transition-all hover:after:w-full">
                    {item.label}
                  </span>
                </a>
              </li>
            ))}
          </ul>

          {/* Mobile dropdown */}
          {menuOpen && (
            <ul
              className="absolute right-0 top-12 w-64 flex-col gap-1 rounded-2xl border border-border bg-background/95 p-2 shadow-sm backdrop-blur md:hidden"
              onClick={() => setMenuOpen(false)}
            >
              {links.map((item) => (
                <li key={item.href}>
                  <a
                    className="block rounded-xl px-4 py-2 text-sm hover:bg-muted/40"
                    href={item.href}
                  >
                    {item.label}
                  </a>
                </li>
              ))}
            </ul>
          )}
        </div>
      </nav>
    </header>
  );
}
