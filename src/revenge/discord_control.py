from __future__ import annotations

import asyncio
import io
import logging

import cv2
import discord
from discord.ext import commands

log = logging.getLogger(__name__)


def create_bot(runtime, owner: int):
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents, owner_id=owner)

    @bot.check
    async def authorized(ctx):
        return ctx.author.id == owner

    @bot.event
    async def on_ready():
        log.info("Discord ready as %s; commands restricted to owner %d", bot.user, owner)

    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, (commands.CheckFailure, commands.CommandNotFound)):
            return
        if isinstance(error, commands.CommandInvokeError):
            error = error.original
        if isinstance(error, (ValueError, commands.UserInputError)):
            await ctx.send(str(error), allowed_mentions=discord.AllowedMentions.none())
        else:
            log.error("Discord command failed", exc_info=error)
            await ctx.send("Command failed. Check the local log.")

    @bot.command()
    async def start(ctx):
        await asyncio.to_thread(runtime.set_running, True)
        await ctx.send("Started.")

    @bot.command()
    async def stop(ctx):
        await asyncio.to_thread(runtime.set_running, False)
        await ctx.send("Paused. Touches released.")

    @bot.command()
    async def restart(ctx):
        runtime.request_restart()
        await ctx.send("Game restart queued.")

    @bot.command()
    async def brawler(ctx, *, name: str):
        await asyncio.to_thread(runtime.set_brawler, name)
        await ctx.send(
            f"Profile: {runtime.info()['brawler']}. Select this brawler in the game too."
        )

    @bot.command(name="status")
    async def status_command(ctx):
        info = await asyncio.to_thread(runtime.info)
        counts = info["results"]
        await ctx.send(
            f"{info['screen']} | {info['brawler']} | "
            f"W {counts['victory']} / L {counts['defeat']} / D {counts['draw']}"
        )

    @bot.command()
    async def screenshot(ctx):
        def encode():
            rgb = runtime.screenshot()
            success, data = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not success:
                raise ValueError("Could not encode screenshot")
            return io.BytesIO(data.tobytes())

        data = await asyncio.to_thread(encode)
        await ctx.send(file=discord.File(data, filename="screenshot.png"))

    return bot


async def serve(runtime, owner, token):
    bot = create_bot(runtime, owner)
    try:
        async with bot:
            worker = asyncio.create_task(asyncio.to_thread(runtime.run))
            connection = asyncio.create_task(bot.start(token))
            try:
                done, _ = await asyncio.wait(
                    (worker, connection), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    await task
            finally:
                runtime.stop.set()
                await asyncio.to_thread(runtime.shutdown)
                connection.cancel()
                await asyncio.gather(connection, return_exceptions=True)
                await worker
    except discord.DiscordException as error:
        raise RuntimeError(f"Discord connection failed: {error}") from error
