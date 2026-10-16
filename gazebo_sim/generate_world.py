#!/usr/bin/env python3
"""
generate_world.py
Generates forest.world for Gazebo Harmonic from a config.
All object Z poses are computed automatically from the heightmap.

Usage:
    python3 generate_world.py
Output:
    forest.world  (same directory as this script)
"""

from PIL import Image
import numpy as np
import os
import random
import subprocess


# Source paths (used for heightmap sampling only)
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MATERIALS_DIR = os.path.join(SCRIPT_DIR, 'materials', 'textures')
#HEIGHTMAP_PNG = os.path.join(MATERIALS_DIR, 'bc_terrain_heightmap_256.png')
HEIGHTMAP_PNG = os.path.join(MATERIALS_DIR, 'bc_terrain_heightmap_257.png')


# Installed paths (written into the world SDF for Gazebo to load)
# Installed paths — container mounts source at same structure, use SCRIPT_DIR
GZ_MATERIALS = os.path.join(SCRIPT_DIR, 'materials', 'textures')
#GZ_HEIGHTMAP = os.path.join(GZ_MATERIALS, 'bc_terrain_heightmap_256.png')
GZ_HEIGHTMAP = os.path.join(GZ_MATERIALS, 'bc_terrain_heightmap_257.png')
GZ_DIFFUSE   = os.path.join(GZ_MATERIALS, 'bc_moss_rock_diffuse.png')
GZ_NORMAL    = os.path.join(GZ_MATERIALS, 'bc_moss_rock_normal.png')
GZ_TERRAIN_MESH = os.path.join(SCRIPT_DIR, 'materials', 'meshes', 'terrain.obj')

TERRAIN_W   = 10.0
TERRAIN_H   = 10.0
TERRAIN_Z   = 1.0
TERRAIN_POS = (0, 0, 0)
TEXTURE_SIZE = 4

X_MIN, X_MAX = -5, 5
Y_MIN, Y_MAX = -5, 5

OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'world', 'cafe.world')

# ─────────────────────────────────────────────
# HEIGHTMAP SAMPLER
# ─────────────────────────────────────────────
img = Image.open(HEIGHTMAP_PNG).convert("L")
arr = np.array(img)
IMG_H, IMG_W = arr.shape

    
TREE_HALF_HEIGHT = 1.0  # half of tree mesh height in metres

def terrain_z(wx, wy):
    #px = int(np.clip((wx + TERRAIN_W/2) / TERRAIN_W * IMG_W, 0, IMG_W-1))
    #py = int(np.clip((wy + TERRAIN_H/2) / TERRAIN_H * IMG_H, 0, IMG_H-1))
    #normalized = float(arr[py, px]) / 255.0
    #return normalized * TERRAIN_Z
    return 0.0
# ─────────────────────────────────────────────
# OBJECTS CONFIG  ← add / remove / move freely
# Each entry: (name, x, y, z_offset, type, ...type_args)
#
# Types:
#   "mesh"    → (uri, scale, collision_radius, collision_length)
#   "include" → (model_uri,)          uses <include>
#   "apriltag"→ (model_uri, roll, pitch, yaw, z_offset_override)
# ─────────────────────────────────────────────
NUM_TREES_SPRUCE = 10
TREES_SPRUCE = []
for i in range(NUM_TREES_SPRUCE):
	x = random.uniform(X_MIN, X_MAX)
	y = random.uniform(Y_MIN, Y_MAX)
	scale = random.uniform(0.5, 0.6)
	entry = ("spruce_tree_"+str(i), x, y, 0, scale)
	TREES_SPRUCE.append(entry)
	
NUM_TREES_FIR = 10
TREES_FIR = []
for i in range(NUM_TREES_FIR):
	x = random.uniform(X_MIN, X_MAX)
	y = random.uniform(Y_MIN, Y_MAX)
	scale = random.uniform(0.15, 0.2)
	entry = ("fir_tree_"+str(i), x, y, 0, scale)
	TREES_FIR.append(entry)



APRILTAGS = [
    # (name,                    x,     y,    z_offset, roll,   pitch,  yaw)
    ("Apriltag36_11_00000",  -4.96,  1.5,   0.46,    1.5708, 0.0,  1.5708),
]



# SDF BUILDERS

def tree_sdf_spruce(name, x, y, z_offset, scale):
    
    tz = terrain_z(x, y)
    z  = tz + z_offset
    s  = scale
    return f"""
    <!-- {name}: terrain_z({x}, {y}) = {tz:.4f} -->
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z:.4f} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>model://spruce_tree/spruce_tree.glb</uri>
              <scale>{s} {s} {s}</scale>
            </mesh>
          </geometry>
        </visual>

      </link>
    </model>"""

def tree_sdf_fir(name, x, y, z_offset, scale):
    
    tz = terrain_z(x, y)
    z  = tz + z_offset
    s  = scale
    return f"""
    <!-- {name}: terrain_z({x}, {y}) = {tz:.4f} -->
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z:.4f} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>model://fir_tree/lhpine.glb</uri>
              <scale>{s} {s} {s}</scale>
            </mesh>
          </geometry>
        </visual>

      </link>
    </model>"""
    
# ─────────────────────────────────────────────
# TERRAIN COLLISION (dartsim can't build heightmap/mesh collision —
# see https://github.com/gazebosim/gz-physics/issues/450 / #451,
# still open. Use a grid of box primitives instead, same trick as
# the tree cylinder collisions above.)
# ─────────────────────────────────────────────
TERRAIN_GRID_N   = 25          # 25x25 cells over the 50x50m terrain -> 2m cells
TERRAIN_BOX_THICK = 3.0        # generous thickness so gentle slope steps never leave a gap
TERRAIN_SURFACE_MARGIN = 0.02  # tiny lift so box top sits just at/above visual surface

def terrain_collision_grid_sdf(grid_n=TERRAIN_GRID_N):
    cell_w = TERRAIN_W / grid_n
    cell_h = TERRAIN_H / grid_n
    boxes = []
    for i in range(grid_n):
        for j in range(grid_n):
            cx = -TERRAIN_W/2 + (i + 0.5) * cell_w
            cy = -TERRAIN_H/2 + (j + 0.5) * cell_h
            tz = terrain_z(cx, cy)
            z  = tz + TERRAIN_SURFACE_MARGIN - TERRAIN_BOX_THICK/2
            boxes.append(f"""
        <collision name="terrain_cell_{i}_{j}">
          <pose>{cx:.4f} {cy:.4f} {z:.4f} 0 0 0</pose>
          <geometry>
            <box>
              <size>{cell_w*1.02:.4f} {cell_h*1.02:.4f} {TERRAIN_BOX_THICK}</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.9</mu>
                <mu2>0.9</mu2>
              </ode>
            </friction>
          </surface>
        </collision>""")

    return f"""
    <model name="terrain_collision_grid">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="link">{"".join(boxes)}
      </link>
    </model>"""
    
def all_tree_collisions_sdf(spruce_trees, fir_trees):
    # One static model, one link, one <collision> per tree — instead of
    # 120 separate models. Collision pose is in world coordinates via the
    # link's identity pose, so we bake terrain_z + z_offset directly in.
    def collision_element(name, x, y, z_offset):
        tz = terrain_z(x, y)
        z = tz + z_offset + 1.0  # +1.0 = half of the 2.0m cylinder length
        return f"""
        <collision name="{name}_collision">
          <pose>{x} {y} {z:.4f} 0 0 0</pose>
          <geometry>
            <cylinder>
              <radius>0.15</radius>
              <length>2.0</length>
            </cylinder>
          </geometry>
        </collision>"""

    all_collisions = "".join(
        collision_element(name, x, y, z_offset)
        for name, x, y, z_offset, scale in spruce_trees + fir_trees
    )

    return f"""
    <model name="forest_tree_collisions">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="link">{all_collisions}
      </link>
    </model>"""

def apriltag_sdf(name, x, y, z_offset, roll, pitch, yaw):
    tz = terrain_z(x, y)
    z  = tz + z_offset
    return f"""
    <!-- {name}: terrain_z({x}, {y}) = {tz:.4f} -->
    <include>
      <name>{name}</name>
      <pose>{x} {y} {z:.4f} {roll} {pitch} {yaw}</pose>
      <static>true</static>
      <uri>model://{name}</uri>
    </include>"""


# ASSEMBLE WORLD

tree_xml_spruce     = "".join(tree_sdf_spruce(*t)     for t in TREES_SPRUCE)
tree_xml_fir     = "".join(tree_sdf_fir(*t)     for t in TREES_FIR)

apriltag_xml = "".join(apriltag_sdf(*t) for t in APRILTAGS)

# addeddd
tree_collisions_xml = all_tree_collisions_sdf(TREES_SPRUCE, TREES_FIR)
terrain_collision_xml = terrain_collision_grid_sdf()

world = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <plugin filename="gz-sim-physics-system"          name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system"    name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <plugin filename="gz-sim-sensors-system"          name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system"              name="gz::sim::systems::Imu"></plugin>
    <plugin name='rgl::RGLServerPluginManager' filename='RGLServerPluginManager'>
        <do_ignore_entities_in_lidar_link>true</do_ignore_entities_in_lidar_link>
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

<model name="terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>{TERRAIN_W} {TERRAIN_H}</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.9</mu>
                <mu2>0.9</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>{TERRAIN_W} {TERRAIN_H}</size>
            </plane>
          </geometry>
          <material>
            <ambient>1 1 1 1</ambient>
            <diffuse>1 1 1 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
            <pbr>
              <metal>
                <albedo_map>{GZ_DIFFUSE}</albedo_map>
                <normal_map>{GZ_NORMAL}</normal_map>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>


{tree_xml_spruce}
{tree_xml_fir}


{tree_collisions_xml}

    <!-- ── AprilTags ─────────────────────────── -->
{apriltag_xml}

  </world>
</sdf>
"""

# Fix paths for container
world = world.replace('/home/administrator/go_sim/src', '/root/ws/src')
with open(OUTPUT_FILE, "w") as f:
    f.write(world)

print(f"Written: {OUTPUT_FILE}")
print(f"\nTerrain: {TERRAIN_W}x{TERRAIN_H}m, Z scale {TERRAIN_Z}m")
print(f"Heightmap sampled at {IMG_W}x{IMG_H}px\n")
print("Object Z poses:")
for name, x, y, z_off, scale in TREES_SPRUCE:
    tz = terrain_z(x, y)
    print(f"  {name:20s}  terrain_z={tz:.4f}  pose_z={tz+z_off:.4f}")
for name, x, y, z_off, scale in TREES_FIR:
    tz = terrain_z(x, y)
    print(f"  {name:20s}  terrain_z={tz:.4f}  pose_z={tz+z_off:.4f}")

for name, x, y, z_off, *_ in APRILTAGS:
    tz = terrain_z(x, y)
    print(f"  {name:20s}  terrain_z={tz:.4f}  pose_z={tz+z_off:.4f}")
