#!/usr/bin/env bash
# Exercises the three headline scenarios against a running server (default http://localhost:8000).
B=${BASE:-http://localhost:8000}
J='{status, outcome, triage:"\(.triage.category)/\(.triage.urgency)", steps:[.steps[]|"\(.node): \(.title)"], ticket:(.ticket|{id,status,resolution}), reply, pending_approval:(.pending_approval|if .==null then null else {amount,order_id,why_approval_needed} end)}'
post() { echo "\$ curl -X POST $B$1 -d '$2'" >&2; curl -s -X POST "$B$1" -H 'Content-Type: application/json' -d "$2"; }
echo "=== 1. Small refund auto-resolves ==="
post /api/chat '{"message":"The USB-C cable from ORD-1001 is defective. Please refund $24.99.","customer_email":"alice@example.com"}' | jq "$J"
echo; echo "=== 2. Big refund pauses, then is approved ==="
R=$(post /api/chat '{"message":"My headphones from ORD-1002 stopped working after a week. I want a full refund.","customer_email":"alice@example.com"}'); echo "$R" | jq "$J"
TID=$(echo "$R" | jq -r .thread_id)
echo; echo "\$ curl $B/api/approvals"; curl -s $B/api/approvals | jq -c ".[]|{thread_id,ticket_id,action,order_id,amount,why_approval_needed}"
echo; post /api/approvals/$TID '{"decision":"approve","note":"Verified defect, within 30 days","approver":"support-lead"}' | jq "$J"
echo; echo "=== 3. Unknown customer escalates ==="
post /api/chat '{"message":"I was charged twice for my last order, please fix this.","customer_email":"ghost@nowhere.com"}' | jq "$J"
echo; echo "\$ curl $B/api/tickets"; curl -s $B/api/tickets | jq -c '.[]|{id,customer_email,category,urgency,status}'
