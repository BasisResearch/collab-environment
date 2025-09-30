#!/bin/bash

SERVER_URL="http://localhost:5052"
BASE_PATH="simulated_data/hackathon"

register_sim() {
    local id="$1"
    local folder="$2"

    curl -s -X POST $SERVER_URL/api/add_simulation \
      -H "Content-Type: application/json" \
      -d "{\"simulation_id\": \"$id\", \"folder_path\": \"$BASE_PATH/$folder\", \"config_path\": \"$BASE_PATH/$folder/config.yaml\"}" \
      | python -c "import sys, json; data=json.load(sys.stdin); print('✅ Registered:', data.get('simulation', {}).get('name', 'Unknown')) if data.get('success') else print('❌ Failed:', data.get('error', 'Unknown error'))"
}

# Register all simulations
echo "🎯 Registering all simulations..."

# Boid small 200 series
register_sim "boid-small-200-align-cohesion" "hackathon-boid-small-200-align-cohesion_sim_run-started-20250926-214330"
register_sim "boid-small-200-align" "hackathon-boid-small-200-align-sim_run-started-20250926-214512"
register_sim "boid-small-200-cohesion" "hackathon-boid-small-200-cohesion_sim_run-started-20250926-214434"
register_sim "boid-small-200-baseline" "hackathon-boid-small-200-sim_run-started-20250926-220926"

# Boid small 400 series
register_sim "boid-small-400-align-cohesion" "hackathon-boid-small-400-align-cohesion_sim_run-started-20250926-211906"
register_sim "boid-small-400-align" "hackathon-boid-small-400-align-sim_run-started-20250926-220719"
register_sim "boid-small-400-cohesion" "hackathon-boid-small-400-cohesion-sim_run-started-20250926-220753"
register_sim "boid-small-400-baseline" "hackathon-boid-small-400-sim_run-started-20250926-220632"

# Homogenous 200-20 series
register_sim "homogenous-200-20-0.0-0.0" "hackathon-homogenous-0.02-200-20-0.0-0.0-sim_run-started-20250927-144818"
register_sim "homogenous-200-20-0.0-0.5" "hackathon-homogenous-0.02-200-20-0.0-0.5-sim_run-started-20250927-145343"
register_sim "homogenous-200-20-1.0-0.0" "hackathon-homogenous-0.02-200-20-1.0-0.0-sim_run-started-20250927-145909"
register_sim "homogenous-200-20-1.0-0.5" "hackathon-homogenous-0.02-200-20-1.0-0.5-sim_run-started-20250927-150434"

# Homogenous 200-80 series
register_sim "homogenous-200-80-0.0-0.0" "hackathon-homogenous-0.02-200-80-0.0-0.0-sim_run-started-20250927-151014"
register_sim "homogenous-200-80-0.0-0.5" "hackathon-homogenous-0.02-200-80-0.0-0.5-sim_run-started-20250927-161012"
register_sim "homogenous-200-80-1.0-0.0" "hackathon-homogenous-0.02-200-80-1.0-0.0-sim_run-started-20250927-161613"
register_sim "homogenous-200-80-1.0-0.5" "hackathon-homogenous-0.02-200-80-1.0-0.5-sim_run-started-20250927-162213"

# Homogenous 400-20 series
register_sim "homogenous-400-20-0.0-0.0" "hackathon-homogenous-0.02-400-20-0.0-0.0-sim_run-started-20250927-152349"
register_sim "homogenous-400-20-0.0-0.5" "hackathon-homogenous-0.02-400-20-0.0-0.5-sim_run-started-20250927-152928"
register_sim "homogenous-400-20-1.0-0.0" "hackathon-homogenous-0.02-400-20-1.0-0.0-sim_run-started-20250927-153454"
register_sim "homogenous-400-20-1.0-0.5" "hackathon-homogenous-0.02-400-20-1.0-0.5-sim_run-started-20250927-154018"

# Homogenous 400-80 series
register_sim "homogenous-400-80-0.0-0.0" "hackathon-homogenous-0.02-400-80-0.0-0.0-sim_run-started-20250927-154539"
register_sim "homogenous-400-80-0.0-0.5" "hackathon-homogenous-0.02-400-80-0.0-0.5-sim_run-started-20250927-155050"
register_sim "homogenous-400-80-1.0-0.0" "hackathon-homogenous-0.02-400-80-1.0-0.0-sim_run-started-20250927-155632"
register_sim "homogenous-400-80-1.0-0.5" "hackathon-homogenous-0.02-400-80-1.0-0.5-sim_run-started-20250927-160200"

# Homogenous random series
register_sim "homogenous-random-200-20" "hackathon-homogenous-0.02-random-200-20-0.0-0.0-sim_run-started-20250927-163131"
register_sim "homogenous-random-200-80" "hackathon-homogenous-0.02-random-200-80-0.0-0.0-sim_run-started-20250927-163638"
register_sim "homogenous-random-400-20" "hackathon-homogenous-0.02-random-400-20-0.0-0.0-sim_run-started-20250927-164151"
register_sim "homogenous-random-400-80" "hackathon-homogenous-0.02-random-400-80-0.0-0.0-sim_run-started-20250927-164657"

# Random series
register_sim "random-1-small-200" "hackathon-random-1-small-200-sim_run-started-20250926-224017"
register_sim "random-1-small-400" "hackathon-random-1-small-400-sim_run-started-20250926-224045"
register_sim "random-1-small-small-200" "hackathon-random-1-small-small-200-sim_run-started-20250927-090951"
register_sim "random-1-small-small-400" "hackathon-random-1-small-small-400-sim_run-started-20250926-233448"

# Small-small 200 series
register_sim "small-small-200-align-cohesion" "hackathon-small-small-200-align-cohesion-sim_run-started-20250927-091248"
register_sim "small-small-200-align" "hackathon-small-small-200-align-sim_run-started-20250927-091140"
register_sim "small-small-200-cohesion" "hackathon-small-small-200-cohesion-sim_run-started-20250927-091225"
register_sim "small-small-200-baseline" "hackathon-small-small-200-sim_run-started-20250927-091051"

# Small-small 400 series
register_sim "small-small-400-align-cohesion" "hackathon-small-small-400-align-cohesion-sim_run-started-20250926-231024"
register_sim "small-small-400-align" "hackathon-small-small-400-align-sim_run-started-20250926-231101"
register_sim "small-small-400-cohesion" "hackathon-small-small-400-cohesion-sim_run-started-20250926-231139"
register_sim "small-small-400-baseline" "hackathon-small-small-400-sim_run-started-20250926-231234"

echo "📊 Current simulations:"
curl -s $SERVER_URL/api/simulations | python -c "import sys, json; sims=json.load(sys.stdin); [print(f'  • {s[\"name\"]} ({s[\"num_episodes\"]} episodes)') for s in sims]"