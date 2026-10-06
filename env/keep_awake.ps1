# Reversible sleep inhibition on one native thread, with checked API returns.
# Display sleep is allowed. No persistent power-plan changes are made.
param(
    [string]$FlagPath = 'D:\minimind\runs\keep_awake.active',
    [string]$StatusPath = 'D:\minimind\runs\keep_awake_status.json',
    [int]$MaxSeconds = 0
)
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
public static class MiniMindPower {
    [DllImport("kernel32.dll")]
    private static extern uint SetThreadExecutionState(uint flags);
    [DllImport("kernel32.dll")]
    private static extern uint GetCurrentThreadId();

    private static void Status(string path, string state, uint previous, int count) {
        string text = "{\"state\":\"" + state + "\",\"utc\":\"" +
            DateTime.UtcNow.ToString("o") + "\",\"pid\":" +
            Process.GetCurrentProcess().Id + ",\"native_thread_id\":" +
            GetCurrentThreadId() + ",\"previous_execution_state\":" + previous +
            ",\"refresh_count\":" + count + "}";
        string temp = path + ".tmp";
        File.WriteAllText(temp, text);
        if (File.Exists(path)) File.Replace(temp, path, null);
        else File.Move(temp, path);
    }

    public static void Run(string flag, string status, int maxSeconds) {
        var timer = Stopwatch.StartNew();
        int count = 0;
        try {
            while (File.Exists(flag) && (maxSeconds <= 0 || timer.Elapsed.TotalSeconds < maxSeconds)) {
                uint previous = SetThreadExecutionState(0x80000001u);
                if (previous == 0) throw new InvalidOperationException("SetThreadExecutionState failed");
                Status(status, "active", previous, ++count);
                // Keep all native calls on this thread and refresh after wake.
                for (int i = 0; i < 15; i++) {
                    if (!File.Exists(flag) || (maxSeconds > 0 && timer.Elapsed.TotalSeconds >= maxSeconds)) break;
                    Thread.Sleep(1000);
                }
            }
        } finally {
            uint previous = SetThreadExecutionState(0x80000000u);
            Status(status, previous == 0 ? "release_failed" : "released", previous, count);
        }
    }
}
'@
[MiniMindPower]::Run($FlagPath, $StatusPath, $MaxSeconds)
