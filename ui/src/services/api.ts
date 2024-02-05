import {
  EventSourceMessage,
  fetchEventSource,
} from "@microsoft/fetch-event-source";

export const BACKEND_BASE_URL = process.env.REACT_APP_BACKEND_HOST || "http://0.0.0.0:8005"


export const fetchAnalyseTweetsSummaryMessage = async (
  // session: string,
  messages: { role: string, content: string }[],
  // model: string,git
  onopen: (res: Response) => void,
  onmessage: (event: EventSourceMessage) => void,
  onerror: (err: any) => void,
  onclose: () => void,
  signal: AbortSignal,
  uids: string
) => {
  await fetchEventSource(`${BACKEND_BASE_URL}/analyse-tweets-event`, {
    openWhenHidden: true,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "access_key": "hello"
    },
    body: JSON.stringify({ messages, uids: uids.split(',').map(uid => parseInt(uid.trim(), 10)) }),
    // @ts-ignore
    onopen(res) {
      onopen(res);
    },
    onmessage(event:any) {
      onmessage(event);
    },
    onclose() {
      onclose();
    },
    onerror(err:any) {
      onerror(err);
    },
    signal,
  });
};
