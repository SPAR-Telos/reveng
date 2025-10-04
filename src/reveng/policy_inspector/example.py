from reveng.agents.alpha_start_agent import AlphaStarAgent
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.policy_inspector.policy_elicitation import elicit_policy, visualize_policy

# Create a simple environment for policy elicitation
print("Creating environment...")
env = Simple2DNavigationEnv(
    size=7,  # Small grid for easier visualization
    complexity=0.3,  # Some walls but not too complex
    render_mode=None,  # No rendering needed for policy elicitation
)
env.reset()

print(f"Environment created with size {env.width}x{env.height}")
print(f"Agent start position: {env.agent_pos}")
print(f"Goal position: {env.goal_pos}")

# Create agents
print("\nCreating agents...")
astar_agent = AlphaStarAgent(name="A*")

# Elicit policies
print("\n" + "=" * 50)
print("Eliciting A* policy...")
print("=" * 50)
astar_policy = elicit_policy(env, astar_agent)

# Visualize and save A* policy as PNG
visualize_policy(
    astar_policy,
    env,
    filename="src/reveng/policy_inspector/policy_maps/astar_policy.png",
    title="A* Agent Policy",
)

print("\nPolicy elicitation complete!")
