"""A replaceable tool loop; receives public task data and local bridge capabilities."""

import json
import os
import urllib.request


def request(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        os.environ["CODING_RL_BRIDGE_URL"] + path,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["CODING_RL_BRIDGE_TOKEN"],
        },
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    with open(os.environ["CODING_RL_TASK"]) as stream:
        task = json.load(stream)
    tools = request("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    schemas = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in tools
    ]
    messages = task["messages"]
    for turn in range(task["max_turns"]):
        result = request(
            "/v1/chat/completions", {"model": "policy", "messages": messages, "tools": schemas}
        )
        choice = result["choices"][0]
        message = choice["message"]
        messages.append(message)
        if choice["finish_reason"] == "length" or not message.get("tool_calls"):
            print(message.get("content", ""))
            return
        for call in message["tool_calls"]:
            function = call["function"]
            result = request(
                "/mcp",
                {
                    "jsonrpc": "2.0",
                    "id": turn + 2,
                    "method": "tools/call",
                    "params": {
                        "name": function["name"],
                        "arguments": json.loads(function["arguments"]),
                    },
                },
            )["result"]
            observation = "\n".join(c["text"] for c in result["content"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})


if __name__ == "__main__":
    main()
