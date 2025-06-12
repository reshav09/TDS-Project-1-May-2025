# TDS Virtual TA (Vercel Deployment)

This project implements a RAG-based virtual TA for the IITM TDS course, combining course content and Discourse posts.

## Deployment

- Built with Flask and OpenAI API (via AIProxy)
- Hosted on Vercel using `vercel-python`

## Environment Variables

- `AIPROXY_TOKEN` — required (set via Vercel's Environment Variables UI)

## Endpoints

- POST `/api/` with:
  ```json
  {
    "question": "...",
    "image": "base64string (optional)"
  }


##Credits
- IIT Madras Data Science
- Developed by Reshav Sharma, June 2025.

