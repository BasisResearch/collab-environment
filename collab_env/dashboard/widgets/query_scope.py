"""
Query scope abstraction for flexible data selection.

Defines what subset of data to analyze (episode, session, or custom filters).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ScopeType(Enum):
    """Type of data scope for analysis."""
    EPISODE = "episode"
    SESSION = "session"
    CUSTOM = "custom"


@dataclass
class QueryScope:
    """
    Defines what data subset to analyze.

    Supports three types of scopes:
    - EPISODE: Single episode with optional time filtering
    - SESSION: All episodes in a session
    - CUSTOM: Arbitrary filters for advanced queries

    Examples
    --------
    >>> # Analyze specific time range in episode
    >>> scope = QueryScope.from_episode("ep_123", start_time=0, end_time=500)

    >>> # Analyze entire session
    >>> scope = QueryScope.from_session("sess_456", agent_type="agent")

    >>> # Custom filtered analysis
    >>> scope = QueryScope.from_custom(
    ...     session_id="sess_456",
    ...     agent_ids=["agent_1", "agent_2"],
    ...     min_speed=5.0
    ... )
    """

    scope_type: ScopeType

    # Identifiers (at least one required)
    episode_id: Optional[str] = None
    session_id: Optional[str] = None

    # Time filtering
    start_time: Optional[int] = None
    end_time: Optional[int] = None

    # Agent filtering
    agent_type: Optional[str] = None  # "agent", "target", "all", or custom
    agent_ids: Optional[List[str]] = None  # Specific agent IDs

    # Arbitrary filters (for custom scope)
    custom_filters: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_episode(
        cls,
        episode_id: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = "agent"
    ) -> "QueryScope":
        """
        Create scope for single episode.

        Parameters
        ----------
        episode_id : str
            Episode identifier
        start_time : int, optional
            Start time (frame number)
        end_time : int, optional
            End time (frame number)
        agent_type : str, default="agent"
            Agent type filter

        Returns
        -------
        QueryScope
            Episode scope
        """
        return cls(
            scope_type=ScopeType.EPISODE,
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type
        )

    @classmethod
    def from_session(
        cls,
        session_id: str,
        agent_type: str = "agent"
    ) -> "QueryScope":
        """
        Create scope for full session (all episodes).

        Parameters
        ----------
        session_id : str
            Session identifier
        agent_type : str, default="agent"
            Agent type filter

        Returns
        -------
        QueryScope
            Session scope
        """
        return cls(
            scope_type=ScopeType.SESSION,
            session_id=session_id,
            agent_type=agent_type
        )

    @classmethod
    def from_custom(cls, **filters) -> "QueryScope":
        """
        Create custom scope with arbitrary filters.

        Parameters
        ----------
        **filters
            Arbitrary filter key-value pairs

        Returns
        -------
        QueryScope
            Custom scope

        Examples
        --------
        >>> scope = QueryScope.from_custom(
        ...     session_id="sess_123",
        ...     agent_ids=["a1", "a2"],
        ...     min_speed=5.0
        ... )
        """
        return cls(
            scope_type=ScopeType.CUSTOM,
            custom_filters=filters
        )

    def to_query_params(self) -> Dict[str, Any]:
        """
        Convert scope to dictionary of query parameters.

        Returns
        -------
        dict
            Query parameters suitable for QueryBackend methods
        """
        params: Dict[str, Any] = {}

        if self.episode_id:
            params['episode_id'] = self.episode_id
        if self.session_id:
            params['session_id'] = self.session_id
        if self.start_time is not None:
            params['start_time'] = self.start_time
        if self.end_time is not None:
            params['end_time'] = self.end_time
        if self.agent_type:
            params['agent_type'] = self.agent_type
        if self.agent_ids:
            params['agent_ids'] = self.agent_ids
        if self.custom_filters:
            params.update(self.custom_filters)

        return params

    def __str__(self) -> str:
        """String representation for debugging."""
        if self.scope_type == ScopeType.EPISODE:
            time_range = ""
            if self.start_time is not None or self.end_time is not None:
                time_range = f" [{self.start_time or 0}:{self.end_time or '∞'}]"
            return f"Episode({self.episode_id}{time_range}, {self.agent_type})"
        elif self.scope_type == ScopeType.SESSION:
            return f"Session({self.session_id}, {self.agent_type})"
        else:
            filters = ", ".join(f"{k}={v}" for k, v in self.custom_filters.items())
            return f"Custom({filters})"
