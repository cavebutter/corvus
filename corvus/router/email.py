"""Deterministic router for the email triage pipeline.

Receives an EmailTriageTask from the classifier executor, applies
confidence gates, and either executes the IMAP action or queues it
for human review.

No LLM calls — pure Python logic.
"""

import logging

from corvus.audit.email_logger import EmailAuditLog
from corvus.integrations.imap import ImapClient
from corvus.queue.email_review import EmailReviewQueue
from corvus.schemas.document_tagging import GateAction
from corvus.schemas.email import EmailActionType, EmailTriageTask

logger = logging.getLogger(__name__)


async def execute_email_action(
    task: EmailTriageTask,
    *,
    imap: ImapClient,
) -> None:
    """Execute the proposed IMAP action for an email triage task.

    Args:
        task: The email triage task with the proposed action.
        imap: An open ImapClient instance.
    """
    action = task.proposed_action

    if action.action_type == EmailActionType.DELETE:
        await imap.delete([task.uid])
    elif action.action_type == EmailActionType.MOVE:
        if action.target_folder:
            await imap.move([task.uid], action.target_folder)
        else:
            logger.warning(
                "MOVE action for email %s has no target folder, skipping",
                task.uid,
            )
    elif action.action_type == EmailActionType.FLAG:
        flag = action.flag_name or "\\Flagged"
        await imap.flag([task.uid], flag)
    elif action.action_type == EmailActionType.MARK_READ:
        await imap.mark_read([task.uid])
    elif action.action_type == EmailActionType.KEEP:
        pass  # no action needed
    else:
        logger.warning("Unknown action type: %s", action.action_type)


async def route_email_action(
    task: EmailTriageTask,
    *,
    imap: ImapClient,
    force_queue: bool = True,
    audit_log: EmailAuditLog,
    review_queue: EmailReviewQueue,
) -> bool:
    """Apply confidence gates and execute or queue an email action.

    Args:
        task: The email triage task from the classifier.
        imap: An open ImapClient instance.
        force_queue: If True, queue all actions for review regardless
            of confidence (initial safe posture).
        audit_log: The email audit log.
        review_queue: The email review queue.

    Returns:
        True if the action was applied, False if queued for review.
    """
    gate = task.gate_action

    # Force queue override: send everything to review
    if force_queue:
        gate = GateAction.QUEUE_FOR_REVIEW

    if gate == GateAction.QUEUE_FOR_REVIEW:
        review_queue.add(task)
        audit_log.log_queued_for_review(task)
        logger.info(
            "Queued email %s for review: %s -> %s (confidence=%.2f)",
            task.uid,
            task.classification.category.value,
            task.proposed_action.action_type.value,
            task.overall_confidence,
        )
        return False

    # AUTO_EXECUTE or FLAG_IN_DIGEST: execute the action
    await execute_email_action(task, imap=imap)
    audit_log.log_auto_applied(task)
    logger.info(
        "Auto-applied email %s: %s -> %s (confidence=%.2f, gate=%s)",
        task.uid,
        task.classification.category.value,
        task.proposed_action.action_type.value,
        task.overall_confidence,
        gate.value,
    )
    return True
