document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // current state
    cstate: {
      time: null,
      messages: [],
    },
    allMessages: [],
    displayMessages: [],

    // historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [],

    home: 0,
    generating: false,
    endpoint: window.location.origin + "/v1",
    model: "llama3-groq-70b-8192-tool-use-preview",
    stopToken: "sfjsdkfjsljflksdkj",

    // performance tracking
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    // New property for error messages
    errorMessage: null,

    // Debug mode
    debug: false,

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => {
        return state.time === cstate.time;
      });
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
    },

    resetChat() {
      this.cstate = {
        time: null,
        messages: [],
      };
      this.allMessages = [];
      this.displayMessages = [];
      this.total_tokens = 0;
      this.time_till_first = 0;
      this.tokens_per_second = 0;
    },

    loadChat(chatState) {
      this.cstate = JSON.parse(JSON.stringify(chatState)); // Deep copy
      this.allMessages = [...this.cstate.messages];
      this.displayMessages = [...this.cstate.messages];
      this.home = 1;
      this.updateTotalTokens(this.allMessages);
    },

    async handleSend() {
      const el = document.getElementById("input-form");
      const value = el.value.trim();
      if (!value) return;

      if (this.generating) return;
      this.generating = true;
      this.errorMessage = null;

      // If it's a new chat or there are no messages, reset the chat
      if (this.home === 0 || this.cstate.messages.length === 0) {
        this.resetChat();
      }

      if (this.home === 0) this.home = 1;

      window.history.pushState({}, "", "/");

      const userMessage = { role: "user", content: value };
      this.cstate.messages.push(userMessage);
      this.allMessages.push(userMessage);
      this.displayMessages.push(userMessage);

      el.value = "";
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";

      const prefill_start = Date.now();
      let start_time = 0;
      let tokens = 0;
      this.tokens_per_second = 0;

      try {
        let currentAssistantMessage = null;

        for await (const chunk of this.openaiChatCompletion(this.allMessages)) {
          if (chunk.role === "function") {
            if (currentAssistantMessage) {
              this.allMessages.push(currentAssistantMessage);
              this.displayMessages.push(currentAssistantMessage);
              currentAssistantMessage = null;
            }
            this.allMessages.push(chunk);
            if (this.debug) {
              this.displayMessages.push(chunk);
            }
          } else {
            if (!currentAssistantMessage) {
              currentAssistantMessage = { role: "assistant", content: "" };
            }
            currentAssistantMessage.content += chunk;

            tokens += 1;
            this.total_tokens += 1;
            if (start_time === 0) {
              start_time = Date.now();
              this.time_till_first = start_time - prefill_start;
            } else {
              const diff = Date.now() - start_time;
              if (diff > 0) {
                this.tokens_per_second = tokens / (diff / 1000);
              }
            }
          }
        }

        // Add any pending assistant message
        if (currentAssistantMessage) {
          this.allMessages.push(currentAssistantMessage);
          this.displayMessages.push(currentAssistantMessage);
        }

        // Update total tokens using all messages
        this.updateTotalTokens(this.allMessages);

        // Update cstate.messages with displayMessages
        this.cstate.messages = [...this.displayMessages];

        const index = this.histories.findIndex(
          (cstate) => cstate.time === this.cstate.time,
        );
        this.cstate.time = Date.now();
        if (index !== -1) {
          this.histories[index] = this.cstate;
        } else {
          this.histories.push(this.cstate);
        }
        localStorage.setItem("histories", JSON.stringify(this.histories));
      } catch (error) {
        console.error("Error in handleSend:", error);
        this.showError(
          error.message || "An error occurred processing your request.",
        );
      } finally {
        this.generating = false;
      }
    },

    async handleEnter(event) {
      if (!event.shiftKey) {
        event.preventDefault();
        await this.handleSend();
      }
    },

    showError(message) {
      this.errorMessage = message;
      setTimeout(() => {
        this.errorMessage = null;
      }, 3000);
    },

    updateTotalTokens(messages) {
      fetch(`${window.location.origin}/v1/tokenizer/count`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: this.allMessages }), // Always use allMessages for token counting
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error("Failed to count tokens");
          }
          return response.json();
        })
        .then((data) => {
          this.total_tokens = data.token_count;
        })
        .catch((error) => {
          console.error("Error updating total tokens:", error);
          this.showError("Failed to update token count. Please try again.");
        });
    },

    async *openaiChatCompletion(messages) {
      try {
        const response = await fetch(`${this.endpoint}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify({
            model: this.model,
            messages: messages,
            stream: true,
            stop: [this.stopToken],
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to fetch");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]") return;

              try {
                const json = JSON.parse(data);
                if (json.choices && json.choices[0].delta) {
                  const delta = json.choices[0].delta;
                  if (delta.role === "function") {
                    // Yield the entire function message
                    yield {
                      role: "function",
                      name: delta.name,
                      content: delta.content,
                    };
                  } else if (delta.content) {
                    yield delta.content;
                  }
                }
              } catch (error) {
                console.error("Error parsing JSON:", error);
              }
            }
          }
        }
      } catch (error) {
        console.error("Error in openaiChatCompletion:", error);
        this.showError(
          error.message ||
            "An error occurred while communicating with the server.",
        );
        throw error;
      }
    },
  }));
});
