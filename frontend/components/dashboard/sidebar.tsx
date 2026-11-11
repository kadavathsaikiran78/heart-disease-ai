"use client";
import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  Activity,
  LayoutDashboard,
  ClipboardList,
  ChartNoAxesCombined,
  FileText,
  Settings,
  HeartPulse,
  CircleHelp,
} from "lucide-react";
const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Patient Assessment", href: "/assessment", icon: ClipboardList },
  { name: "Model Analytics", href: "/analytics", icon: ChartNoAxesCombined },
  { name: "Prediction Reports", href: "/reports", icon: FileText },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col border-r border-slate-200 bg-white px-5 py-6">
      <Link href="/" className="mb-10 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white">
          <HeartPulse size={23} />
        </div>
        <div>
          <h1 className="font-bold tracking-tight text-slate-900">
            CardioAI
          </h1>
          <p className="text-xs text-slate-500">
            Clinical Analytics
          </p>
        </div>
      </Link>

      <p className="mb-3 px-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Workspace
      </p>

      <nav className="flex-1 space-y-2">
        {navigation.map((item) => {
          const Icon = item.icon;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium transition ${
                pathname === item.href
              }`}
            >
              <Icon size={19} />
              {item.name}
            </Link>
          );
        })}
      </nav>

      <div className="space-y-2 border-t border-slate-200 pt-5">
        <Link
          href="#"
          className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
        >
          <Settings size={18} />
          Settings
        </Link>

        <Link
          href="#"
          className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
        >
          <CircleHelp size={18} />
          Help & Support
        </Link>
      </div>

      <div className="mt-5 rounded-xl bg-slate-50 p-3">
        <div className="flex items-center gap-2">
          <Activity size={17} className="text-emerald-600" />
          <span className="text-xs font-semibold text-slate-700">
            Development Mode
          </span>
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-500">
          Academic research prototype. Predictions use the saved model and are
          not medical diagnoses.
        </p>
      </div>
    </aside>
  );
}
