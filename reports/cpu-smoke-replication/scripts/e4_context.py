"""E4: sliding-window image eviction and prompt-cache prefix stability (paper Sec 2.5).

Grows a synthetic Responses-style history (each tool result carries 2 images) and runs the
real SlidingWindowPolicy each turn. Measures images retained, estimated tokens and how
often the previous request is an exact prefix of the next (a proxy for prompt-cache reuse),
comparing eviction batch sizes.
"""
import json

from uni_robot_agent.context.history import ContextRequest
from uni_robot_agent.context.policies.sliding_window import SlidingWindowPolicy

IMG = "data:image/png;base64," + "A" * 40_000  # ~40 KB per image


def history(turns):
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Stack the bowls."}]}]
    for i in range(turns):
        items.append({"type": "function_call", "call_id": f"c{i}", "name": "move", "arguments": json.dumps({"i": i})})
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": [
            {"type": "input_text", "text": f"feedback {i}: pose_reached"},
            {"type": "input_image", "image_url": IMG},
            {"type": "input_image", "image_url": IMG},
        ]})
    return {"model": "m", "input": items}


def count_images(body):
    return sum(1 for it in body["input"] if it.get("type") == "function_call_output"
               for b in it["output"] if b.get("type") == "input_image")


def run(batch, turns=60):
    pol = SlidingWindowPolicy(max_estimated_input_tokens=10_000_000, target_estimated_input_tokens=9_000_000,
                              min_recent_tool_images=3, image_eviction_batch_size=batch)
    prev, hits, tokens, imgs = None, 0, [], []
    for t in range(1, turns + 1):
        d = pol.select(ContextRequest.from_dict(history(t)))
        items = d.request.body["input"]
        if prev is not None and items[:len(prev)] == prev:
            hits += 1
        prev = items
        tokens.append(d.estimated_tokens_after)
        imgs.append(count_images(d.request.body))
    return hits / (turns - 1), max(imgs), sum(imgs) / len(imgs), max(tokens), sum(tokens) / len(tokens)


def main():
    print(f"{'batch':>5s} {'prefix_reuse%':>13s} {'max_imgs':>8s} {'mean_imgs':>9s} {'max_tok':>9s} {'mean_tok':>9s}")
    for b in (1, 5, 20, 10_000):
        r, mi, ai, mt, at = run(b)
        label = "none" if b == 10_000 else str(b)
        print(f"{label:>5s} {100*r:13.1f} {mi:8d} {ai:9.1f} {mt:9d} {at:9.0f}")


if __name__ == "__main__":
    main()
