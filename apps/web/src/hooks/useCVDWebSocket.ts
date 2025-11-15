/**
 * CVD WebSocket Hook
 *
 * Provides real-time updates for CVD runs via WebSocket connection.
 *
 * Features:
 * - Auto-reconnect on disconnect
 * - Event type filtering
 * - Connection state management
 * - Type-safe event handling
 */

import { useEffect, useRef, useState, useCallback } from "react";

export type CVDEventType =
  | "run_started"
  | "progress_update"
  | "metrics_update"
  | "warning"
  | "error"
  | "run_completed"
  | "run_failed"
  | "run_cancelled"
  | "thickness_update"
  | "stress_risk"
  | "adhesion_risk"
  | "rate_anomaly";

export interface CVDEvent {
  run_id: string;
  event_type: CVDEventType;
  timestamp: string;
  data: Record<string, any>;
}

export interface CVDWebSocketOptions {
  runId: string;
  onEvent?: (event: CVDEvent) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  autoReconnect?: boolean;
  reconnectDelay?: number; // ms
  baseUrl?: string;
}

export type ConnectionState = "connecting" | "connected" | "disconnected" | "error";

export function useCVDWebSocket({
  runId,
  onEvent,
  onConnect,
  onDisconnect,
  onError,
  autoReconnect = true,
  reconnectDelay = 3000,
  baseUrl = "ws://localhost:8001",
}: CVDWebSocketOptions) {
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [lastEvent, setLastEvent] = useState<CVDEvent | null>(null);
  const [events, setEvents] = useState<CVDEvent[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setConnectionState("connecting");

    try {
      const ws = new WebSocket(`${baseUrl}/ws/cvd/runs/${runId}`);

      ws.onopen = () => {
        console.log(`[CVD WebSocket] Connected to run: ${runId}`);
        setConnectionState("connected");
        reconnectAttemptsRef.current = 0;
        onConnect?.();
      };

      ws.onmessage = (message) => {
        try {
          const data = JSON.parse(message.data);

          // Handle different message types
          if (data.type === "event") {
            const event: CVDEvent = {
              run_id: runId,
              event_type: data.event_type,
              timestamp: data.timestamp,
              data: data.data,
            };

            setLastEvent(event);
            setEvents((prev) => [...prev, event]);
            onEvent?.(event);
          } else if (data.type === "connected") {
            console.log(`[CVD WebSocket] Welcome message:`, data.message);
          }
        } catch (error) {
          console.error("[CVD WebSocket] Failed to parse message:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("[CVD WebSocket] Error:", error);
        setConnectionState("error");
        onError?.(error);
      };

      ws.onclose = () => {
        console.log(`[CVD WebSocket] Disconnected from run: ${runId}`);
        setConnectionState("disconnected");
        onDisconnect?.();

        // Auto-reconnect
        if (autoReconnect) {
          const delay = Math.min(reconnectDelay * Math.pow(2, reconnectAttemptsRef.current), 30000);
          console.log(`[CVD WebSocket] Reconnecting in ${delay}ms...`);

          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error("[CVD WebSocket] Connection failed:", error);
      setConnectionState("error");
    }
  }, [runId, baseUrl, autoReconnect, reconnectDelay, onConnect, onDisconnect, onError, onEvent]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState("disconnected");
  }, []);

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn("[CVD WebSocket] Cannot send - not connected");
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    connectionState,
    lastEvent,
    events,
    connect,
    disconnect,
    send,
    isConnected: connectionState === "connected",
  };
}

/**
 * Hook to subscribe to specific event types
 */
export function useCVDEventFilter(
  events: CVDEvent[],
  eventTypes: CVDEventType[]
): CVDEvent[] {
  return events.filter((event) => eventTypes.includes(event.event_type));
}

/**
 * Hook to get latest event of specific type
 */
export function useLatestCVDEvent(
  events: CVDEvent[],
  eventType: CVDEventType
): CVDEvent | null {
  const filtered = events.filter((event) => event.event_type === eventType);
  return filtered.length > 0 ? filtered[filtered.length - 1] : null;
}

/**
 * Example usage:
 *
 * ```tsx
 * function RunMonitor({ runId }: { runId: string }) {
 *   const { connectionState, lastEvent, events, isConnected } = useCVDWebSocket({
 *     runId,
 *     onEvent: (event) => {
 *       console.log("New event:", event);
 *     },
 *   });
 *
 *   // Filter progress updates
 *   const progressEvents = useCVDEventFilter(events, ["progress_update"]);
 *   const latestProgress = useLatestCVDEvent(events, "progress_update");
 *
 *   // Filter warnings and errors
 *   const alerts = useCVDEventFilter(events, ["warning", "error", "stress_risk", "adhesion_risk"]);
 *
 *   return (
 *     <div>
 *       <div>Status: {connectionState}</div>
 *       {latestProgress && (
 *         <div>Progress: {latestProgress.data.progress}%</div>
 *       )}
 *       <div>Total events: {events.length}</div>
 *       <div>Alerts: {alerts.length}</div>
 *     </div>
 *   );
 * }
 * ```
 */
