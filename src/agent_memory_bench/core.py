"""Core benchmark logic -- task definitions and evaluation."""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import (
    BenchmarkReport,
    MemoryEntry,
    Query,
    SampleResult,
    TaskDefinition,
    TaskResult,
    TaskSample,
    TaskType,
)


def compute_recall_at_k(
    retrieved: list[str], expected: list[str], k: int = 5
) -> float:
    """Compute recall@k: fraction of expected items found in top-k retrieved."""
    if not expected:
        return 1.0
    top_k = retrieved[:k]
    found = sum(1 for e in expected if any(e.lower() in r.lower() for r in top_k))
    return found / len(expected)


def compute_mrr(retrieved: list[str], expected: list[str]) -> float:
    """Compute Mean Reciprocal Rank over expected items."""
    if not expected:
        return 1.0
    reciprocal_ranks = []
    for exp in expected:
        for rank, ret in enumerate(retrieved, 1):
            if exp.lower() in ret.lower():
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def generate_fact_recall_task(num_samples: int = 20, seed: int = 42) -> TaskDefinition:
    """Generate a fact recall benchmark task.

    Tests whether the memory system can store facts and retrieve them
    when queried with related questions.
    """
    rng = random.Random(seed)
    facts = [
        ("The capital of France is Paris.", "What is the capital of France?", "Paris"),
        ("Python was created by Guido van Rossum in 1991.", "Who created Python?", "Guido van Rossum"),
        ("The speed of light is approximately 299,792,458 meters per second.", "What is the speed of light?", "299,792,458"),
        ("Mount Everest is 8,849 meters tall.", "How tall is Mount Everest?", "8,849"),
        ("The human body has 206 bones.", "How many bones does the human body have?", "206"),
        ("Water boils at 100 degrees Celsius at sea level.", "At what temperature does water boil?", "100"),
        ("The Great Wall of China is approximately 21,196 kilometers long.", "How long is the Great Wall of China?", "21,196"),
        ("DNA stands for deoxyribonucleic acid.", "What does DNA stand for?", "deoxyribonucleic acid"),
        ("The Amazon River is the largest river by volume.", "What is the largest river by volume?", "Amazon"),
        ("Jupiter is the largest planet in our solar system.", "What is the largest planet?", "Jupiter"),
        ("The first Moon landing was in 1969.", "When was the first Moon landing?", "1969"),
        ("Oxygen has the atomic number 8.", "What is the atomic number of oxygen?", "8"),
        ("Shakespeare wrote 37 plays.", "How many plays did Shakespeare write?", "37"),
        ("The Pacific Ocean is the largest ocean.", "What is the largest ocean?", "Pacific"),
        ("Einstein published the theory of general relativity in 1915.", "When was general relativity published?", "1915"),
        ("The Sahara is the largest hot desert.", "What is the largest hot desert?", "Sahara"),
        ("A marathon is 42.195 kilometers.", "How long is a marathon?", "42.195"),
        ("The Earth is approximately 4.5 billion years old.", "How old is the Earth?", "4.5 billion"),
        ("Beethoven composed 9 symphonies.", "How many symphonies did Beethoven compose?", "9"),
        ("Gold has the chemical symbol Au.", "What is the chemical symbol for gold?", "Au"),
    ]

    samples = []
    selected = rng.sample(facts, min(num_samples, len(facts)))
    for i, (fact, question, answer) in enumerate(selected):
        # Add distractor memories alongside the target fact
        distractors = rng.sample([f for f, _, _ in facts if f != fact], min(3, len(facts) - 1))
        memories = [
            MemoryEntry(content=fact, metadata={"type": "target"}),
        ] + [
            MemoryEntry(content=d, metadata={"type": "distractor"})
            for d in distractors
        ]
        samples.append(
            TaskSample(
                sample_id=f"fact-recall-{i:03d}",
                memories_to_store=memories,
                query=Query(text=question),
                expected_answer=answer,
                expected_retrieved_contents=[fact],
            )
        )

    return TaskDefinition(
        task_type=TaskType.FACT_RECALL,
        name="Fact Recall",
        description="Tests whether the memory system can store facts and retrieve them accurately.",
        samples=samples,
    )


def generate_temporal_reasoning_task(num_samples: int = 10, seed: int = 42) -> TaskDefinition:
    """Generate a temporal reasoning benchmark task.

    Tests whether the memory system handles time-ordered information correctly.
    Each scenario presents an old value and a newer update; the memory system
    should return the most recent value when queried.
    """
    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1)

    temporal_scenarios = [
        (
            "User set their favorite color to blue.",
            "User changed their favorite color to green.",
            "What is the user's current favorite color?",
            "green",
        ),
        (
            "User's timezone is set to PST.",
            "User moved to EST timezone.",
            "What timezone is the user in?",
            "EST",
        ),
        (
            "Default language is English.",
            "Changed default language to Spanish.",
            "What is the default language?",
            "Spanish",
        ),
        (
            "User's email is alice@old.com.",
            "User updated email to alice@new.com.",
            "What is the user's email?",
            "alice@new.com",
        ),
        (
            "Team standup is at 9am daily.",
            "Team standup moved to 10:30am daily.",
            "When is the team standup?",
            "10:30am",
        ),
        (
            "The project uses Python 3.9.",
            "The project was upgraded to Python 3.12.",
            "What Python version does the project use?",
            "Python 3.12",
        ),
        (
            "The deploy target is us-east-1.",
            "Deploy target changed to eu-west-1.",
            "What is the deploy target region?",
            "eu-west-1",
        ),
        (
            "The database password rotates monthly.",
            "Database password rotation changed to weekly.",
            "How often does the database password rotate?",
            "weekly",
        ),
        (
            "The CI pipeline runs on Jenkins.",
            "CI pipeline migrated to GitHub Actions.",
            "What CI system does the project use?",
            "GitHub Actions",
        ),
        (
            "The on-call rotation starts on Mondays.",
            "On-call rotation changed to start on Wednesdays.",
            "When does the on-call rotation start?",
            "Wednesdays",
        ),
        (
            "Max upload file size is 10 MB.",
            "Max upload file size increased to 50 MB.",
            "What is the max upload file size?",
            "50 MB",
        ),
        (
            "The sprint duration is 2 weeks.",
            "Sprint duration changed to 3 weeks.",
            "How long is the sprint?",
            "3 weeks",
        ),
    ]

    samples = []
    selected = rng.sample(temporal_scenarios, min(num_samples, len(temporal_scenarios)))
    for i, (old_stmt, new_stmt, question, answer) in enumerate(selected):
        events = [
            MemoryEntry(
                content=old_stmt,
                timestamp=base_time + timedelta(days=i * 10),
                metadata={"type": "preference", "version": 1},
            ),
            MemoryEntry(
                content=new_stmt,
                timestamp=base_time + timedelta(days=i * 10 + 5),
                metadata={"type": "preference", "version": 2},
            ),
        ]
        rng.shuffle(events)  # Store in random order

        samples.append(
            TaskSample(
                sample_id=f"temporal-{i:03d}",
                memories_to_store=events,
                query=Query(text=question),
                expected_answer=answer,
                expected_retrieved_contents=[new_stmt],
            )
        )

    return TaskDefinition(
        task_type=TaskType.TEMPORAL_REASONING,
        name="Temporal Reasoning",
        description="Tests whether the memory system correctly handles time-ordered updates.",
        samples=samples,
    )


def generate_contradiction_detection_task(num_samples: int = 10, seed: int = 42) -> TaskDefinition:
    """Generate a contradiction detection benchmark task.

    Tests whether the memory system can detect and handle contradictory information.
    """
    samples = []
    contradictions = [
        (
            "The project deadline is March 15th.",
            "The project deadline has been moved to April 1st.",
            "What is the project deadline?",
            "April 1st",
        ),
        (
            "The meeting is scheduled for 2pm.",
            "The meeting has been rescheduled to 4pm.",
            "When is the meeting?",
            "4pm",
        ),
        (
            "The budget for Q1 is $50,000.",
            "The Q1 budget has been revised to $75,000.",
            "What is the Q1 budget?",
            "$75,000",
        ),
        (
            "John is the team lead for Project Alpha.",
            "Sarah has replaced John as team lead for Project Alpha.",
            "Who is the team lead for Project Alpha?",
            "Sarah",
        ),
        (
            "The server runs on Python 3.9.",
            "The server has been upgraded to Python 3.12.",
            "What Python version does the server run?",
            "Python 3.12",
        ),
        (
            "The office is located on the 5th floor.",
            "The team has moved to the 8th floor.",
            "What floor is the office on?",
            "8th floor",
        ),
        (
            "The API rate limit is 100 requests per minute.",
            "The API rate limit has been increased to 500 requests per minute.",
            "What is the API rate limit?",
            "500 requests per minute",
        ),
        (
            "The database uses PostgreSQL 14.",
            "We migrated the database to PostgreSQL 16.",
            "What version of PostgreSQL is the database?",
            "PostgreSQL 16",
        ),
        (
            "The deploy target is AWS us-east-1.",
            "We switched the deploy target to AWS eu-west-1.",
            "What is the deploy target region?",
            "eu-west-1",
        ),
        (
            "The sprint duration is 2 weeks.",
            "We changed sprint duration to 3 weeks.",
            "How long is the sprint?",
            "3 weeks",
        ),
    ]

    for i, (old_fact, new_fact, question, answer) in enumerate(contradictions[:num_samples]):
        base_time = datetime(2026, 1, 1)
        samples.append(
            TaskSample(
                sample_id=f"contradiction-{i:03d}",
                memories_to_store=[
                    MemoryEntry(
                        content=old_fact,
                        timestamp=base_time,
                        metadata={"type": "original"},
                    ),
                    MemoryEntry(
                        content=new_fact,
                        timestamp=base_time + timedelta(days=7),
                        metadata={"type": "update"},
                    ),
                ],
                query=Query(text=question),
                expected_answer=answer,
                expected_retrieved_contents=[new_fact],
            )
        )

    return TaskDefinition(
        task_type=TaskType.CONTRADICTION_DETECTION,
        name="Contradiction Detection",
        description="Tests whether the memory system handles contradictory/updated information.",
        samples=samples,
    )


def generate_cross_conversation_task(num_samples: int = 10, seed: int = 42) -> TaskDefinition:
    """Generate a cross-conversation continuity benchmark task.

    Tests whether the memory system can retrieve information stored in one
    conversation session when queried from a different session.
    """
    rng = random.Random(seed)
    scenarios = [
        (
            ("session-a", "My favorite programming language is Rust."),
            ("session-b", "I also enjoy writing Go for backend services."),
            ("session-c", "What programming language does the user prefer?"),
            "Rust",
            "My favorite programming language is Rust.",
        ),
        (
            ("session-work", "I have a meeting with the design team at 3pm on Tuesday."),
            ("session-personal", "My dentist appointment is Thursday at 10am."),
            ("session-work", "When is the design team meeting?"),
            "3pm on Tuesday",
            "I have a meeting with the design team at 3pm on Tuesday.",
        ),
        (
            ("session-1", "The project uses a PostgreSQL database hosted on AWS RDS."),
            ("session-2", "We decided to add Redis for caching."),
            ("session-3", "What database does the project use?"),
            "PostgreSQL",
            "The project uses a PostgreSQL database hosted on AWS RDS.",
        ),
        (
            ("session-alpha", "Alice's phone number is 555-0123."),
            ("session-beta", "Bob's email is bob@example.com."),
            ("session-gamma", "What is Alice's phone number?"),
            "555-0123",
            "Alice's phone number is 555-0123.",
        ),
        (
            ("session-morning", "The deployment pipeline takes about 12 minutes to complete."),
            ("session-afternoon", "We should optimize the test stage to run in parallel."),
            ("session-evening", "How long does the deployment pipeline take?"),
            "12 minutes",
            "The deployment pipeline takes about 12 minutes to complete.",
        ),
        (
            ("session-jan", "Q1 revenue target is $2 million."),
            ("session-feb", "Marketing budget for Q1 is $200,000."),
            ("session-mar", "What is the Q1 revenue target?"),
            "$2 million",
            "Q1 revenue target is $2 million.",
        ),
        (
            ("session-dev", "The API authentication uses JWT tokens with RS256."),
            ("session-ops", "Rate limiting is set to 1000 requests per minute per client."),
            ("session-security", "What authentication method does the API use?"),
            "JWT tokens with RS256",
            "The API authentication uses JWT tokens with RS256.",
        ),
        (
            ("session-onboard", "New employees get 20 days of PTO per year."),
            ("session-hr", "The health insurance plan is through Blue Cross."),
            ("session-question", "How many PTO days do new employees get?"),
            "20 days",
            "New employees get 20 days of PTO per year.",
        ),
        (
            ("session-frontend", "The UI framework is React 18 with TypeScript."),
            ("session-backend", "The backend is built with FastAPI and Python 3.12."),
            ("session-review", "What UI framework does the project use?"),
            "React 18",
            "The UI framework is React 18 with TypeScript.",
        ),
        (
            ("session-planning", "Sprint velocity has averaged 34 story points."),
            ("session-retro", "The team agreed to limit WIP to 3 items per person."),
            ("session-standup", "What is the team's average sprint velocity?"),
            "34 story points",
            "Sprint velocity has averaged 34 story points.",
        ),
        (
            ("session-research", "The competitor launched a similar feature last quarter."),
            ("session-product", "Our differentiator is real-time collaboration support."),
            ("session-exec", "What is our product differentiator?"),
            "real-time collaboration support",
            "Our differentiator is real-time collaboration support.",
        ),
        (
            ("session-setup", "The CI server runs on a self-hosted GitHub Actions runner."),
            ("session-debug", "Build failures are usually caused by flaky integration tests."),
            ("session-infra", "Where does the CI server run?"),
            "self-hosted GitHub Actions runner",
            "The CI server runs on a self-hosted GitHub Actions runner.",
        ),
    ]

    samples = []
    selected = rng.sample(scenarios, min(num_samples, len(scenarios)))
    for i, (mem1, mem2, query_info, answer, expected_content) in enumerate(selected):
        session1, content1 = mem1
        session2, content2 = mem2
        query_session, query_text = query_info
        base_time = datetime(2026, 1, 1)
        samples.append(
            TaskSample(
                sample_id=f"cross-conv-{i:03d}",
                memories_to_store=[
                    MemoryEntry(
                        content=content1,
                        session_id=session1,
                        timestamp=base_time,
                        metadata={"type": "target"},
                    ),
                    MemoryEntry(
                        content=content2,
                        session_id=session2,
                        timestamp=base_time + timedelta(hours=1),
                        metadata={"type": "distractor"},
                    ),
                ],
                query=Query(text=query_text, session_id=query_session),
                expected_answer=answer,
                expected_retrieved_contents=[expected_content],
            )
        )

    return TaskDefinition(
        task_type=TaskType.CROSS_CONVERSATION,
        name="Cross-Conversation Continuity",
        description="Tests whether the memory system can retrieve information across conversation boundaries.",
        samples=samples,
    )


def generate_multi_hop_retrieval_task(num_samples: int = 10, seed: int = 42) -> TaskDefinition:
    """Generate a multi-hop retrieval benchmark task.

    Tests whether the memory system can retrieve multiple related facts needed
    to answer a question requiring compositional reasoning.
    """
    rng = random.Random(seed)
    chains = [
        (
            "Alice works at Acme Corp.",
            "Acme Corp headquarters is in New York City.",
            "Where is Alice's company headquartered?",
            "New York City",
        ),
        (
            "Bob's manager is Carol.",
            "Carol reports directly to the CTO.",
            "Who does Bob's manager report to?",
            "the CTO",
        ),
        (
            "The app is deployed on Kubernetes cluster 'prod-east'.",
            "Cluster 'prod-east' runs on AWS in us-east-1.",
            "What cloud region is the app deployed in?",
            "us-east-1",
        ),
        (
            "Project Phoenix uses the Spark framework.",
            "The Spark framework requires Java 17 or higher.",
            "What Java version does Project Phoenix require?",
            "Java 17",
        ),
        (
            "The payments service calls the Stripe API.",
            "The Stripe API key is stored in AWS Secrets Manager.",
            "Where is the payments service API key stored?",
            "AWS Secrets Manager",
        ),
        (
            "Dr. Smith is the lead researcher on the vaccine trial.",
            "The vaccine trial is funded by the Gates Foundation.",
            "Who funds Dr. Smith's research?",
            "the Gates Foundation",
        ),
        (
            "The Berlin office handles European sales.",
            "European sales grew 23% last quarter.",
            "How much did the Berlin office's sales grow?",
            "23%",
        ),
        (
            "The mobile app is built with Flutter.",
            "Flutter apps compile to native ARM code.",
            "What does the mobile app compile to?",
            "native ARM code",
        ),
        (
            "Sarah leads the data engineering team.",
            "The data engineering team maintains the Snowflake warehouse.",
            "Who maintains the Snowflake warehouse?",
            "Sarah",
        ),
        (
            "The analytics dashboard uses Grafana.",
            "Grafana pulls metrics from the Prometheus server on port 9090.",
            "What port does the analytics data source run on?",
            "9090",
        ),
        (
            "The CI pipeline is defined in the monorepo.",
            "The monorepo is hosted on GitHub under the 'platform' org.",
            "Where is the CI pipeline hosted?",
            "GitHub under the 'platform' org",
        ),
        (
            "Customer support uses Zendesk for ticketing.",
            "Zendesk is integrated with Slack channel #support-alerts.",
            "Where do customer support alerts go?",
            "Slack channel #support-alerts",
        ),
    ]

    samples = []
    selected = rng.sample(chains, min(num_samples, len(chains)))
    for i, (fact1, fact2, question, answer) in enumerate(selected):
        other_facts = [f1 for f1, _, _, _ in chains if f1 != fact1]
        distractor = rng.choice(other_facts)
        memories = [
            MemoryEntry(content=fact1, metadata={"type": "chain-link-1"}),
            MemoryEntry(content=fact2, metadata={"type": "chain-link-2"}),
            MemoryEntry(content=distractor, metadata={"type": "distractor"}),
        ]
        rng.shuffle(memories)
        samples.append(
            TaskSample(
                sample_id=f"multi-hop-{i:03d}",
                memories_to_store=memories,
                query=Query(text=question),
                expected_answer=answer,
                expected_retrieved_contents=[fact1, fact2],
            )
        )

    return TaskDefinition(
        task_type=TaskType.MULTI_HOP_RETRIEVAL,
        name="Multi-Hop Retrieval",
        description="Tests whether the memory system can retrieve multiple facts needed for compositional reasoning.",
        samples=samples,
    )


def generate_memory_update_task(num_samples: int = 10, seed: int = 42) -> TaskDefinition:
    """Generate a memory update benchmark task.

    Tests whether the memory system can handle explicit updates to stored facts,
    returning the most recent value rather than the original.
    """
    rng = random.Random(seed)
    updates = [
        (
            "User's home address is 123 Oak Street, Springfield.",
            "User updated their home address to 456 Maple Avenue, Shelbyville.",
            "What is the user's home address?",
            "456 Maple Avenue, Shelbyville",
        ),
        (
            "The default branch is 'develop'.",
            "Updated: the default branch has been changed to 'main'.",
            "What is the default branch?",
            "main",
        ),
        (
            "The team standup is at 9:00 AM daily.",
            "Update: standup time changed to 9:30 AM daily.",
            "When is the team standup?",
            "9:30 AM",
        ),
        (
            "The primary contact for vendor X is John at john@vendor.com.",
            "Updated: the primary contact for vendor X is now Lisa at lisa@vendor.com.",
            "Who is the primary contact for vendor X?",
            "Lisa",
        ),
        (
            "The application log level is set to DEBUG.",
            "Updated: log level changed to WARNING for production.",
            "What is the application log level?",
            "WARNING",
        ),
        (
            "The monthly storage quota is 500 GB.",
            "Updated: monthly storage quota increased to 2 TB.",
            "What is the monthly storage quota?",
            "2 TB",
        ),
        (
            "The on-call rotation is weekly, starting on Mondays.",
            "Updated: on-call rotation changed to bi-weekly, starting on Wednesdays.",
            "How does the on-call rotation work?",
            "bi-weekly, starting on Wednesdays",
        ),
        (
            "The staging environment URL is staging.example.com.",
            "Updated: staging environment moved to stage.newdomain.io.",
            "What is the staging environment URL?",
            "stage.newdomain.io",
        ),
        (
            "Password policy requires minimum 8 characters.",
            "Updated: password policy now requires minimum 12 characters with special characters.",
            "What is the password policy minimum length?",
            "12 characters",
        ),
        (
            "The data retention period is 30 days.",
            "Updated: data retention period extended to 90 days.",
            "What is the data retention period?",
            "90 days",
        ),
        (
            "Max file upload size is 10 MB.",
            "Updated: max file upload size increased to 50 MB.",
            "What is the max file upload size?",
            "50 MB",
        ),
        (
            "The backup schedule runs at midnight UTC.",
            "Updated: backup schedule changed to run at 3 AM UTC.",
            "When do backups run?",
            "3 AM UTC",
        ),
    ]

    samples = []
    selected = rng.sample(updates, min(num_samples, len(updates)))
    for i, (original, updated, question, answer) in enumerate(selected):
        base_time = datetime(2026, 1, 1)
        samples.append(
            TaskSample(
                sample_id=f"mem-update-{i:03d}",
                memories_to_store=[
                    MemoryEntry(
                        content=original,
                        timestamp=base_time,
                        metadata={"type": "original", "version": 1},
                    ),
                    MemoryEntry(
                        content=updated,
                        timestamp=base_time + timedelta(days=14),
                        metadata={"type": "update", "version": 2, "supersedes": "original"},
                    ),
                ],
                query=Query(text=question),
                expected_answer=answer,
                expected_retrieved_contents=[updated],
            )
        )

    return TaskDefinition(
        task_type=TaskType.MEMORY_UPDATE,
        name="Memory Update",
        description="Tests whether the memory system correctly handles explicit updates and returns the latest value.",
        samples=samples,
    )


def generate_context_window_efficiency_task(num_samples: int = 10, seed: int = 42) -> TaskDefinition:
    """Generate a context window efficiency benchmark task.

    Tests how well the memory system maintains retrieval quality as the number
    of stored memories grows from small to large.
    """
    rng = random.Random(seed)
    target_facts = [
        ("The company was founded in 2015.", "When was the company founded?", "2015"),
        ("The CEO's name is Maria Chen.", "Who is the CEO?", "Maria Chen"),
        ("Annual revenue last year was $4.2 million.", "What was last year's revenue?", "$4.2 million"),
        ("The main product is an AI-powered code review tool.", "What is the main product?", "AI-powered code review tool"),
        ("The engineering team has 28 members.", "How large is the engineering team?", "28"),
        ("The next board meeting is on April 10th.", "When is the next board meeting?", "April 10th"),
        ("The primary customer segment is mid-market SaaS companies.", "Who is the primary customer segment?", "mid-market SaaS companies"),
        ("The server uptime SLA is 99.95%.", "What is the uptime SLA?", "99.95%"),
        ("The mobile app has 150,000 monthly active users.", "How many monthly active users does the mobile app have?", "150,000"),
        ("The NPS score last quarter was 72.", "What was the NPS score?", "72"),
        ("The data center is located in Frankfurt, Germany.", "Where is the data center?", "Frankfurt, Germany"),
        ("The company has 3 pending patents.", "How many pending patents does the company have?", "3"),
    ]

    filler_templates = [
        "Internal memo #{n}: Discussed progress on sprint goals for week {w}.",
        "Meeting notes #{n}: Reviewed design specs for feature {w}.",
        "Status update #{n}: Completed {w} out of 10 planned tasks.",
        "Team sync #{n}: Addressed blockers for milestone {w}.",
        "Daily log #{n}: Processed {w} support tickets today.",
        "Reminder #{n}: Code freeze for release {w} starts Friday.",
        "Note #{n}: Updated documentation for module {w}.",
        "Action item #{n}: Follow up on feedback from review {w}.",
    ]

    scales = [10, 50, 100, 200]
    samples = []
    selected_facts = rng.sample(target_facts, min(num_samples, len(target_facts)))

    for i, (fact, question, answer) in enumerate(selected_facts):
        scale = scales[i % len(scales)]
        filler_rng = random.Random(seed + i)

        fillers = []
        for n in range(scale):
            template = filler_rng.choice(filler_templates)
            fillers.append(
                MemoryEntry(
                    content=template.format(n=n, w=filler_rng.randint(1, 100)),
                    metadata={"type": "filler", "scale": scale},
                )
            )

        target_pos = filler_rng.randint(0, len(fillers))
        fillers.insert(
            target_pos,
            MemoryEntry(content=fact, metadata={"type": "target", "scale": scale}),
        )

        samples.append(
            TaskSample(
                sample_id=f"ctx-window-{i:03d}-scale-{scale}",
                memories_to_store=fillers,
                query=Query(text=question),
                expected_answer=answer,
                expected_retrieved_contents=[fact],
                metadata={"scale": scale},
            )
        )

    return TaskDefinition(
        task_type=TaskType.CONTEXT_WINDOW_EFFICIENCY,
        name="Context Window Efficiency",
        description="Tests retrieval accuracy as the number of stored memories grows.",
        samples=samples,
    )


TASK_GENERATORS = {
    TaskType.FACT_RECALL: generate_fact_recall_task,
    TaskType.TEMPORAL_REASONING: generate_temporal_reasoning_task,
    TaskType.CONTRADICTION_DETECTION: generate_contradiction_detection_task,
    TaskType.CROSS_CONVERSATION: generate_cross_conversation_task,
    TaskType.MULTI_HOP_RETRIEVAL: generate_multi_hop_retrieval_task,
    TaskType.MEMORY_UPDATE: generate_memory_update_task,
    TaskType.CONTEXT_WINDOW_EFFICIENCY: generate_context_window_efficiency_task,
}


class BenchmarkRunner:
    """Runs benchmark tasks against registered memory adapters."""

    def __init__(self, seed: int = 42, num_samples: int = 20):
        self._adapters: dict[str, MemoryAdapter] = {}
        self._seed = seed
        self._num_samples = num_samples

    def register_adapter(self, name: str, adapter: MemoryAdapter) -> None:
        """Register a memory adapter for benchmarking."""
        self._adapters[name] = adapter

    @property
    def available_tasks(self) -> list[TaskType]:
        """List of task types with implemented generators."""
        return list(TASK_GENERATORS.keys())

    def run_task(self, adapter: MemoryAdapter, task: TaskDefinition) -> TaskResult:
        """Run a single benchmark task against an adapter."""
        sample_results: list[SampleResult] = []

        for sample in task.samples:
            adapter.clear()
            try:
                # Store memories
                store_start = time.perf_counter()
                for entry in sample.memories_to_store:
                    adapter.store(entry)
                store_elapsed = (time.perf_counter() - store_start) * 1000

                # Retrieve
                retrieve_start = time.perf_counter()
                retrieved = adapter.retrieve(sample.query)
                retrieve_elapsed = (time.perf_counter() - retrieve_start) * 1000

                retrieved_contents = [r.content for r in retrieved]
                recall = compute_recall_at_k(
                    retrieved_contents, sample.expected_retrieved_contents
                )
                mrr = compute_mrr(retrieved_contents, sample.expected_retrieved_contents)

                sample_results.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        retrieved=retrieved,
                        recall_at_k=recall,
                        mrr=mrr,
                        latency_ms=retrieve_elapsed,
                        store_latency_ms=store_elapsed,
                    )
                )
            except Exception as e:
                sample_results.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        retrieved=[],
                        success=False,
                        error=str(e),
                    )
                )

        result = TaskResult(
            task_type=task.task_type,
            adapter_name=adapter.name,
            sample_results=sample_results,
        )
        result.compute_aggregates()
        return result

    def run(self, tasks: list[str | TaskType] | None = None) -> BenchmarkReport:
        """Run benchmarks for the specified tasks across all registered adapters."""
        if not self._adapters:
            raise ValueError("No adapters registered. Call register_adapter() first.")

        # Resolve task types
        task_types: list[TaskType] = []
        if tasks is None:
            task_types = self.available_tasks
        else:
            for t in tasks:
                if isinstance(t, str):
                    task_types.append(TaskType(t))
                else:
                    task_types.append(t)

        report = BenchmarkReport()
        for adapter_name, adapter in self._adapters.items():
            adapter.setup()
            try:
                for task_type in task_types:
                    generator = TASK_GENERATORS.get(task_type)
                    if generator is None:
                        continue
                    task_def = generator(
                        num_samples=self._num_samples, seed=self._seed
                    )
                    result = self.run_task(adapter, task_def)
                    report.task_results.append(result)
            finally:
                adapter.teardown()

        return report
