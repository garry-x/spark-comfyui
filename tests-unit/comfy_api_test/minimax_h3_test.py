import pytest

from comfy_api_nodes.apis.minimax import (
    MinimaxH3ContentItem,
    MinimaxH3ImageUrl,
    MinimaxH3TaskResultResponse,
    MinimaxH3VideoGenerationRequest,
    MinimaxH3VideoGenerationResponse,
)
from comfy_api_nodes import nodes_minimax


def test_h3_text_to_video_request_matches_v2_contract():
    request = MinimaxH3VideoGenerationRequest(
        content=[MinimaxH3ContentItem(type="text", text="A paper boat sails across a pond.")],
        duration=5,
        resolution="2K",
        ratio="16:9",
    )

    assert request.model_dump(exclude_none=True) == {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A paper boat sails across a pond."}],
        "duration": 5,
        "resolution": "2K",
        "ratio": "16:9",
    }


@pytest.mark.asyncio
async def test_h3_node_runs_text_to_video_request_through_task_download(monkeypatch):
    calls = []

    async def fake_sync_op(cls, endpoint, *, response_model, data):
        calls.append((endpoint.path, endpoint.method, data.model_dump(exclude_none=True)))
        return response_model(task_id="h3-task")

    async def fake_poll_op(cls, endpoint, *, response_model, status_extractor, **kwargs):
        calls.append((endpoint.path, endpoint.method, kwargs["poll_interval"]))
        response = response_model(task={"status": "succeeded", "content": {"url": "https://example.com/h3.mp4"}})
        assert status_extractor(response) == "succeeded"
        return response

    async def fake_download(url):
        assert url == "https://example.com/h3.mp4"
        return "video-bytes"

    monkeypatch.setattr(nodes_minimax, "sync_op", fake_sync_op)
    monkeypatch.setattr(nodes_minimax, "poll_op", fake_poll_op)
    monkeypatch.setattr(nodes_minimax, "download_url_to_video_output", fake_download)

    output = await nodes_minimax.MinimaxH3VideoNode.execute(
        prompt_text="A paper boat sails across a pond.",
        ratio="16:9",
        duration=5,
        resolution="2K",
    )

    assert output.result == ("video-bytes",)
    assert calls == [
        (
            "/proxy/minimax/v2/video_generation",
            "POST",
            {
                "model": "MiniMax-H3",
                "content": [{"type": "text", "text": "A paper boat sails across a pond."}],
                "duration": 5,
                "resolution": "2K",
                "ratio": "16:9",
            },
        ),
        ("/proxy/minimax/v2/query/video_generation/h3-task", "GET", 10.0),
    ]


def test_h3_image_item_serializes_first_frame_role():
    item = MinimaxH3ContentItem(
        type="image_url",
        image_url=MinimaxH3ImageUrl(url="https://example.com/frame.png"),
        role="first_frame",
    )

    assert item.model_dump(exclude_none=True) == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/frame.png"},
        "role": "first_frame",
    }
