import asyncio, pytz, datetime as dt
from typing import List, Dict
from src.unipile_client import UnipileClient
from src.personalize import craft_messages
from src.transform import Profile
from src.logging_conf import logger

class CampaignStats: sent=0; errors=0

async def run_campaign(profiles: List[Profile], followup_days=(3,7,14)) -> CampaignStats:
    client = UnipileClient()          # uses env vars
    stats = CampaignStats()
    utc = pytz.UTC
    for p in profiles:
        try:
            # 1 fetch enriched data
            pdata = await client.get_profile(p.linkedin_url)
            posts = await client.recent_posts(pdata["id"], limit=1)
            recent = posts[0]["text"] if posts else ""
            msgs = craft_messages(pdata, recent)

            # 2 send invitation
            await client.send_invitation(pdata["id"], msgs["connection"])

            # 3 (optional) comment
            if posts:
                await client.comment_post(posts[0]["id"], msgs["comment"])

            # 4 schedule follow‑ups
            # we ask Unipile delayed-sends
            conv_id = pdata["id"]     # simplification; real conv id comes after accept
            now = dt.datetime.utcnow().replace(tzinfo=utc)
            for i,txt in enumerate(msgs["followups"]):
                send_at = (now + dt.timedelta(days=followup_days[i])).isoformat()
                await client.send_message(conv_id, txt, send_at=send_at)

            stats.sent += 1
        except Exception as e:
            stats.errors +=1
            logger.error(f"Campaign error for {p.linkedin_url}: {e}")
    return stats 