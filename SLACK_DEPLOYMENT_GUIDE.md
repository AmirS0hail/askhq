# Slack Bot Deployment Guide — Render

This guide walks through deploying the AskHQ Slack bot to Render's free tier.

**Prerequisites:**
- ✅ Slack bot is working locally
- ✅ Backend API is live at: `https://askhq.onrender.com`
- ✅ You have your Slack bot tokens (from Slack App dashboard)

---

## Step 1: Create Render Account

1. Go to [render.com](https://render.com)
2. Click **"Get Started"** or **"Sign Up"**
3. Sign up with GitHub (recommended) or email
4. **No credit card required** for free tier
5. Verify your email if prompted

---

## Step 2: Connect GitHub Repository

1. From Render dashboard, click **"New +"** → **"Web Service"**
2. Click **"Connect account"** under GitHub
3. Authorize Render to access your GitHub repos
4. Select your **Slack bot repository** from the list
5. Click **"Connect"**

---

## Step 3: Configure the Web Service

### Basic Settings

| Field | Value |
|---|---|
| **Name** | `askhq-slack-bot` (or your preferred name) |
| **Region** | Oregon (or closest to you) |
| **Branch** | `main` (or whatever branch has your Slack bot code) |
| **Root Directory** | Leave blank (unless bot is in a subdirectory) |

### Build & Deploy Settings

**If you have a Dockerfile:**
- **Runtime**: Docker
- Build/Start commands: Auto-detected from Dockerfile

**If you're using Python directly:**
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py` (or whatever starts your Slack bot)

### Instance Type
- **Plan**: **Free** (512 MB RAM)

---

## Step 4: Set Environment Variables

Scroll down to **"Environment Variables"** and add these:

| Key | Value | Notes |
|---|---|---|
| `SLACK_BOT_TOKEN` | `xoxb-...` | From Slack App → OAuth & Permissions |
| `SLACK_APP_TOKEN` | `xapp-...` | From Slack App → Basic Information → App-Level Tokens |
| `SLACK_SIGNING_SECRET` | `...` | From Slack App → Basic Information |
| `ASKHQ_API_URL` | `https://askhq.onrender.com/ask` | **The FastAPI backend we just deployed** |
| `PORT` | `3000` | (Render will override this, but set as fallback) |

**Where to find Slack tokens:**
1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Select your AskHQ app
3. **Bot Token**: OAuth & Permissions → Bot User OAuth Token (`xoxb-...`)
4. **App Token**: Basic Information → App-Level Tokens → Generate if needed (`xapp-...`)
5. **Signing Secret**: Basic Information → App Credentials → Signing Secret

---

## Step 5: Deploy

1. Click **"Create Web Service"** at the bottom
2. Render will:
   - Clone your repo
   - Install dependencies
   - Start your Slack bot
   - Assign a public URL (e.g., `https://askhq-slack-bot.onrender.com`)

3. Watch the **live logs** for:
   - ✅ Success: `⚡️ Bolt app is running!` or similar
   - ❌ Errors: Missing env vars, port binding issues, etc.

---

## Step 6: Update Slack App Settings

Once deployed, you need to update Slack to point to your Render URL:

### If using **Slack Events API** (HTTP endpoint):

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → your app
2. Click **Event Subscriptions**
3. **Request URL**: `https://YOUR-RENDER-URL.onrender.com/slack/events`
4. Slack will send a challenge request — your bot must respond (if code is correct, it'll auto-verify)
5. Save changes

### If using **Socket Mode** (WebSocket):

- No URL update needed! Socket Mode connects outbound from Render to Slack.
- Make sure `SLACK_APP_TOKEN` is set correctly.

---

## Step 7: Test the Deployment

### Test in Slack:

1. Go to your Slack workspace
2. Invite the bot to a channel: `/invite @AskHQ`
3. Ask a question:
   ```
   @AskHQ How many casual leave days do I get?
   ```

4. Bot should respond with the answer from the RAG backend!

### Check Render Logs:

- Open your service in Render dashboard
- Click **"Logs"** tab
- You should see:
  - Incoming Slack events
  - API calls to `https://askhq.onrender.com/ask`
  - Responses sent back to Slack

---

## Troubleshooting

| Issue | Solution |
|---|---|
| **"Bot doesn't respond"** | Check Render logs for errors. Verify env vars are set correctly. |
| **"Slack event verification failed"** | Make sure `SLACK_SIGNING_SECRET` is correct. Check logs for the exact error. |
| **"Connection timeout to backend"** | Verify `ASKHQ_API_URL=https://askhq.onrender.com/ask` (no trailing slash). Test the backend directly: `curl https://askhq.onrender.com/health` |
| **"Service crashes on startup"** | Check logs for missing dependencies. Make sure `requirements.txt` includes all Slack SDK packages. |
| **"Free tier sleeps"** | Free tier sleeps after 15 min of inactivity. First request after sleep takes ~30-60s. This is normal. |

---

## Free Tier Limits

- **RAM**: 512 MB
- **Sleep**: After 15 min inactivity (first request wakes it, ~30-60s delay)
- **Monthly hours**: 750 (enough for 1 service always-on)
- **Builds**: Unlimited

**Cold start behavior**: When the service wakes from sleep, the Slack bot reconnects to Slack automatically (if using Socket Mode) or handles the next webhook when it arrives.

---

## Next Steps

Once deployed and verified:
1. Share the Slack bot with your team
2. Add more policy documents to the backend's `knowledge/` folder
3. Re-run the backend's `/reindex` endpoint to update the knowledge base:
   ```bash
   curl -X POST https://askhq.onrender.com/reindex \
     -H "X-Reindex-Token: reindex-secret-xyz123"
   ```

---

**Questions?** Check the Render logs first — they show all errors with stack traces. If stuck, share the error message from the logs.

**Backend API**: `https://askhq.onrender.com/ask`  
**Health check**: `https://askhq.onrender.com/health`

Good luck! 🚀
