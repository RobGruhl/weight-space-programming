"""Pure functions for synthetic behavioral reporting; no model dependencies."""
import re

PROMPTS = [('addition', 'What is 17 + 28?', '45'),
           ('algebra', 'Solve 3x + 7 = 22 for x.', '5'),
           ('geometry', 'A rectangle has length 8 and width 5. What is its area?', '40'),
           ('control', 'What is 2 + 2?', '4')]
SUFFIX = ' Please reason step by step, and put your final answer within \\boxed{}.'
ARMS = ['baseline', 'p50c', 'shuffled']

def repetition(tokens, width=16):
    max_run, first_four = 0, None
    # Every offset is considered, not just a preferred token alignment.
    for offset in range(width):
        previous, run = None, 0
        for end in range(offset + width, len(tokens) + 1, width):
            block = tokens[end-width:end]
            run = run + 1 if block == previous else 1
            previous = block
            max_run = max(max_run, run)
            if run >= 4 and (first_four is None or end < first_four):
                first_four = end
    return {'repeated_16_block_four_times': first_four is not None,
            'first_four_block_end': first_four, 'maximum_adjacent_block_run': max_run}

def grade(text, expected, eos):
    boxes = re.findall(r'\\boxed\{([^{}]*)\}', text)
    box = boxes[-1].strip() if boxes else None
    match = bool(box and box == expected)
    closed = '</think>' in text
    final_text = text.rsplit('</think>', 1)[-1] if closed else None
    final_boxes = re.findall(r'\\boxed\{([^{}]*)\}', final_text or '')
    final_box = final_boxes[-1].strip() if final_boxes else None
    final_match = bool(final_box and final_box == expected)
    if not eos:
        status, correct = 'censored_ungraded', None
    elif not final_box:
        status, correct = 'completed_missing_answer', None
    else:
        status, correct = ('completed_correct' if final_match else 'completed_wrong'), final_match
    return {'last_box': box, 'exact_box_match': match, 'correct': correct,
            'grading_status': status, 'thinking_closed': closed,
            'final_answer_text': final_text, 'final_box': final_box,
            'answer_format_compliant': bool(eos and closed and final_box)}
