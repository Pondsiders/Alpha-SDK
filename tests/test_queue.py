"""Tests for queue.py — async message queue."""

import asyncio

import pytest

from alpha_sdk.queue import Message, MessageQueue


# -- Message dataclass --------------------------------------------------------


class TestMessage:
    def test_basic_construction(self):
        msg = Message(content="hello")
        assert msg.content == "hello"
        assert msg.producer == ""

    def test_with_producer(self):
        msg = Message(content="hello", producer="human")
        assert msg.producer == "human"

    def test_empty_content(self):
        msg = Message(content="")
        assert msg.content == ""


# -- MessageQueue basics ------------------------------------------------------


class TestMessageQueueBasics:
    @pytest.fixture
    def queue(self):
        return MessageQueue()

    def test_starts_empty(self, queue):
        assert queue.empty
        assert queue.qsize == 0
        assert not queue.closed

    def test_starts_open(self, queue):
        assert not queue.closed

    @pytest.mark.asyncio
    async def test_put_and_get(self, queue):
        msg = Message(content="hello", producer="test")
        await queue.put(msg)
        assert queue.qsize == 1

        result = await queue.get()
        assert result is not None
        assert result.content == "hello"
        assert result.producer == "test"
        assert queue.empty

    @pytest.mark.asyncio
    async def test_fifo_order(self, queue):
        await queue.put(Message(content="first"))
        await queue.put(Message(content="second"))
        await queue.put(Message(content="third"))

        assert (await queue.get()).content == "first"
        assert (await queue.get()).content == "second"
        assert (await queue.get()).content == "third"

    @pytest.mark.asyncio
    async def test_put_nowait(self, queue):
        msg = Message(content="quick")
        queue.put_nowait(msg)
        assert queue.qsize == 1

        result = await queue.get()
        assert result.content == "quick"

    @pytest.mark.asyncio
    async def test_get_nowait(self, queue):
        await queue.put(Message(content="ready"))
        result = queue.get_nowait()
        assert result is not None
        assert result.content == "ready"

    @pytest.mark.asyncio
    async def test_get_nowait_empty_raises(self, queue):
        with pytest.raises(asyncio.QueueEmpty):
            queue.get_nowait()


# -- Closing and draining ----------------------------------------------------


class TestMessageQueueClose:
    @pytest.fixture
    def queue(self):
        return MessageQueue()

    @pytest.mark.asyncio
    async def test_close_sets_flag(self, queue):
        await queue.close()
        assert queue.closed

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, queue):
        await queue.close()
        await queue.close()  # Should not raise or double-sentinel
        assert queue.closed

    @pytest.mark.asyncio
    async def test_put_after_close_raises(self, queue):
        await queue.close()
        with pytest.raises(RuntimeError, match="closed"):
            await queue.put(Message(content="too late"))

    @pytest.mark.asyncio
    async def test_put_nowait_after_close_raises(self, queue):
        await queue.close()
        with pytest.raises(RuntimeError, match="closed"):
            queue.put_nowait(Message(content="too late"))

    @pytest.mark.asyncio
    async def test_close_sends_sentinel(self, queue):
        """Consumer gets None after close — the shutdown signal."""
        await queue.close()
        result = await queue.get()
        assert result is None

    @pytest.mark.asyncio
    async def test_messages_before_close_are_delivered(self, queue):
        """Messages put before close() are delivered before the sentinel."""
        await queue.put(Message(content="first"))
        await queue.put(Message(content="second"))
        await queue.close()

        assert (await queue.get()).content == "first"
        assert (await queue.get()).content == "second"
        assert (await queue.get()) is None  # Sentinel

    @pytest.mark.asyncio
    async def test_drain(self, queue):
        await queue.put(Message(content="a"))
        await queue.put(Message(content="b"))

        messages = await queue.drain()
        assert len(messages) == 2
        assert messages[0].content == "a"
        assert messages[1].content == "b"
        assert queue.empty

    @pytest.mark.asyncio
    async def test_drain_excludes_sentinel(self, queue):
        await queue.put(Message(content="keep"))
        await queue.close()

        messages = await queue.drain()
        # Should get the real message but not the sentinel
        contents = [m.content for m in messages]
        assert "keep" in contents


# -- Backpressure (bounded queue) --------------------------------------------


class TestMessageQueueBackpressure:
    @pytest.mark.asyncio
    async def test_bounded_queue_blocks_when_full(self):
        queue = MessageQueue(maxsize=1)
        await queue.put(Message(content="fills it"))

        # Second put should block — verify with a timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                queue.put(Message(content="blocks")),
                timeout=0.05,
            )

    @pytest.mark.asyncio
    async def test_bounded_queue_unblocks_after_get(self):
        queue = MessageQueue(maxsize=1)
        await queue.put(Message(content="first"))

        async def delayed_get():
            await asyncio.sleep(0.02)
            return await queue.get()

        # Start a get that will free space, then put should succeed
        get_task = asyncio.create_task(delayed_get())
        await queue.put(Message(content="second"))  # Should unblock
        result = await get_task
        assert result.content == "first"

        result2 = await queue.get()
        assert result2.content == "second"

    @pytest.mark.asyncio
    async def test_put_nowait_full_raises(self):
        queue = MessageQueue(maxsize=1)
        queue.put_nowait(Message(content="fills it"))
        with pytest.raises(asyncio.QueueFull):
            queue.put_nowait(Message(content="too much"))


# -- Multiple producers ------------------------------------------------------


class TestMultipleProducers:
    @pytest.mark.asyncio
    async def test_two_producers(self):
        queue = MessageQueue()

        async def producer(name: str, count: int):
            for i in range(count):
                await queue.put(Message(content=f"{name}-{i}", producer=name))

        await asyncio.gather(
            producer("alpha", 3),
            producer("beta", 3),
        )
        await queue.close()

        messages = []
        while True:
            msg = await queue.get()
            if msg is None:
                break
            messages.append(msg)

        assert len(messages) == 6
        # Both producers contributed
        producers = {m.producer for m in messages}
        assert producers == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_consumer_loop_pattern(self):
        """The standard consumer pattern: get until None."""
        queue = MessageQueue()

        # Simulate a producer that sends then closes
        async def produce():
            for i in range(5):
                await queue.put(Message(content=str(i)))
            await queue.close()

        asyncio.create_task(produce())

        received = []
        while True:
            msg = await queue.get()
            if msg is None:
                break
            received.append(msg.content)

        assert received == ["0", "1", "2", "3", "4"]
