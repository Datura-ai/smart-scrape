export interface Image {
  url: string;
}

export interface Message {
  author: "user" | "assistant" | "system";
  text: string;
  type: "text" | "image" | "image-loading" | "text-loading" | "error";
  content?: string;
  role?: string;
  images?: Image[];
  metadata?: { images: Image[], size?: number; completion?: any; type: "text" | "image" | "image-loading" | "text-loading" | "error"; };
  uuid?: string;
};