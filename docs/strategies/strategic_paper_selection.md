# Strategic Paper Selection for HADES Dataset

## Pivot Strategy: Focus on Accessible High-Conveyance Sources

### 1. Modern Anthropology/STS Papers (More Accessible)

**Target Areas:**
- Digital anthropology
- Computational ethnography  
- Science and Technology Studies (STS)
- Human-Computer Interaction (HCI) anthropology
- Algorithmic culture studies

**Why These Work:**
- Published 2000-2024 (more likely open access)
- Already bridge-aware (discuss technology)
- Often in CS venues (CHI, CSCW) = accessible
- Authors understand both domains

**Search Keywords:**
- "algorithmic anthropology"
- "digital ethnography machine learning"
- "computational culture"
- "AI anthropology"
- "data science ethnography"

### 2. ML Papers as Conveyance Validators

**Conveyance Metrics from Papers:**
```python
conveyance_indicators = {
    "benchmark_improvement": 0.0-1.0,  # How much SOTA improved
    "citations_per_month": count,       # Adoption rate
    "implementations": count,           # GitHub repos
    "blog_posts": count,               # Explanations
    "time_to_first_impl": days,       # Speed of adoption
}
```

**High Conveyance Examples:**
- Transformer paper (2017) → 1000s of implementations
- BERT (2018) → Entire ecosystem in months
- GPT papers → Immediate widespread adoption

**Low Conveyance Examples:**
- Marginal improvement papers → Few implementations
- Complex papers without code → Slow adoption
- Theory-heavy papers → Long discussion, little action

### 3. Code-with-Papers Analysis

**Three-Layer Analysis:**

**Layer 1: Paper Metrics**
- Clarity of algorithm description
- Pseudocode presence
- Mathematical completeness
- Reproducibility details

**Layer 2: Implementation Metrics**
- Time to first GitHub implementation
- Number of independent implementations
- Code quality/completeness
- Documentation quality

**Layer 3: Production Metrics**
- Integration into libraries (HuggingFace, etc.)
- Commercial adoption
- Community improvements
- Real-world applications

### 4. Building the Dataset

**Phase 1: Recent Anthropology/STS (500 papers)**
- Digital anthropology (2010-2024)
- Computational ethnography
- STS engaging with AI/ML
- Open access preferred

**Phase 2: ML Benchmark Papers (1000 papers)**
- Papers with clear benchmark results
- Track: initial scores → community improvements
- Measure: paper → code → adoption timeline
- Focus on 2017-2024 (modern era)

**Phase 3: Code-with-Papers Projects (500 projects)**
- Papers with official implementations
- Active GitHub communities
- Multiple implementations of same paper
- Production deployments

### 5. Conveyance Measurement Framework

**For Anthropology/STS Papers:**
```python
conveyance_score = {
    "has_method": 0.0-1.0,          # Actionable methodology?
    "has_data": 0.0-1.0,            # Shared datasets?
    "has_code": 0.0-1.0,            # Any implementation?
    "bridges_domains": 0.0-1.0,      # Connects anthro-CS?
}
```

**For ML Papers:**
```python
conveyance_score = {
    "benchmark_delta": improvement,   # SOTA improvement
    "implementation_count": n,        # GitHub repos
    "adoption_speed": days,          # Paper → widespread use
    "explanation_count": n,          # Blogs, videos, tutorials
}
```

**For Code Projects:**
```python
conveyance_score = {
    "paper_clarity": 0.0-1.0,        # How clear is theory?
    "code_completeness": 0.0-1.0,    # Matches paper claims?
    "community_size": n,             # Stars, forks, contributors
    "production_use": 0.0-1.0,       # Real-world deployment?
}
```

### 6. Expected Insights

**Hypothesis Validation:**
1. Anthropology papers without methods/code → Low conveyance → Little impact
2. ML papers with big benchmark gains → High conveyance → Rapid adoption
3. Code quality correlates with paper clarity → Better theory enables practice
4. Time-to-implementation inversely proportional to conveyance score

**Bridge Identification:**
- Papers citing both anthropology AND showing benchmarks
- STS papers that include implementable methods
- ML papers addressing cultural/social questions
- Code projects implementing anthropological concepts

### 7. Practical Next Steps

1. **Scrape recent STS/digital anthropology** (Google Scholar, open venues)
2. **Collect ML papers with benchmark tables** (ArXiv, Papers with Code)
3. **Analyze code-with-papers repositories** (GitHub API)
4. **Build conveyance scoring system** based on these metrics
5. **Validate**: High conveyance papers should have more impact

This approach gives us:
- Accessible data (recent, open)
- Measurable outcomes (benchmarks, implementations)
- Clear theory-practice bridges
- Evidence for the conveyance hypothesis