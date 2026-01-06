# Workflow

## Development Cycle

### Task Execution
1. Read and understand the task specification
2. Implement changes following code styleguides
3. Write/update tests for changed functionality
4. Run tests and verify passing
5. Commit changes with descriptive message

### Commit Strategy
- **Frequency:** Commit after each task completion
- **Message Format:** `type(scope): description`
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
  - Example: `feat(tts): add multilingual support for French`

### Git Notes
- Use git notes for task summaries and context
- Attach notes to commits with implementation details
- Format: Brief summary + key decisions made

## Quality Standards

### Test Coverage
- **Target:** >80% code coverage
- Focus on public API methods
- Include edge cases and error paths

### Code Review Checklist
- [ ] Follows PEP 8 and project style guides
- [ ] Type hints on public functions
- [ ] Docstrings on public classes/methods
- [ ] Tests added/updated
- [ ] No new linter warnings
- [ ] Error messages are actionable

## Phase Completion

### Manual Verification Protocol
At the end of each phase:
1. Review all completed tasks
2. Run full test suite
3. Check for regressions
4. Update documentation if needed
5. Tag phase completion in git

### Definition of Done
- Code implemented and tested
- Documentation updated
- No blocking issues
- Reviewed against spec
