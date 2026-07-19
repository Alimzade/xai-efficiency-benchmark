# Agent Rules

- Do not provide single-line git commits. Commit messages must always include a concise subject line followed by a list of main changes using dashes `-`.
- Do not run git add, git commit, or git push commands directly on the user's system. Always output the exact git commands as plain text in the chat for the user to run.
- Do not provide git commit commands or draft commit messages in your responses unless the user explicitly and personally requests it.
- When writing commit messages, do not include bullets detailing intermediate/temporary changes or debug/revert iterations that occurred during the session. Only list the net main changes relative to the last committed state.


