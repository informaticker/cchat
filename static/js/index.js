document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // current state
    cstate: {
      time: null,
      messages: [],
    },

    // historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [],

    home: 0,
    generating: false,
    endpoint: window.location.origin + "/v1",
    model: "llama3-8b-8192", // This doesn't matter anymore as the backend handles it now
    stopToken: "<|eot_id|>", // We may need this for some models

    // performance tracking
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    // New property for error messages
    errorMessage: null,

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => {
        return state.time === cstate.time;
      });
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
    },

    async handleSend() {
      const el = document.getElementById("input-form");
      const value = el.value.trim();
      if (!value) return;

      if (this.generating) return;
      this.generating = true;
      this.errorMessage = null; // Clear any previous error messages
      if (this.home === 0) this.home = 1;

      // ensure that going back in history will go back to home
      window.history.pushState({}, "", "/");

      // add message to list
      this.cstate.messages.push({ role: "user", content: value });

      // clear textarea
      el.value = "";
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";

      // reset performance tracking
      const prefill_start = Date.now();
      let start_time = 0;
      let tokens = 0;
      this.tokens_per_second = 0;

      try {
        // start receiving server sent events
        let gottenFirstChunk = false;
        for await (const chunk of this.openaiChatCompletion(
          this.cstate.messages,
        )) {
          if (!gottenFirstChunk) {
            this.cstate.messages.push({ role: "assistant", content: "" });
            gottenFirstChunk = true;
          }

          // add chunk to the last message
          this.cstate.messages[this.cstate.messages.length - 1].content +=
            chunk;

          // calculate performance tracking
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

        // update the state in histories or add it if it doesn't exist
        const index = this.histories.findIndex((cstate) => {
          return cstate.time === this.cstate.time;
        });
        this.cstate.time = Date.now();
        if (index !== -1) {
          // update the time
          this.histories[index] = this.cstate;
        } else {
          this.histories.push(this.cstate);
        }
        // update in local storage
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
      // if shift is not pressed
      if (!event.shiftKey) {
        event.preventDefault();
        await this.handleSend();
      }
    },

    showError(message) {
      this.errorMessage = message;
      setTimeout(() => {
        this.errorMessage = null;
      }, 3000); // Hide after 5 seconds
    },

    updateTotalTokens(messages) {
      fetch(`${window.location.origin}/v1/tokenizer/count`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages }),
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
                if (json.choices && json.choices[0].delta.content) {
                  yield json.choices[0].delta.content;
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
