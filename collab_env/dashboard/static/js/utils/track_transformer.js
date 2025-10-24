/**
 * Track Transformer - Applies manual coordinate transformations to 3D tracks
 *
 * Provides translation, rotation, and uniform scaling transformations
 * for aligning 3D track data with mesh geometry.
 */

import * as THREE from 'three';

export class TrackTransformer {
    constructor() {
        // Transformation parameters
        this.translation = { x: 0, y: 0, z: 0 };
        this.rotation = { x: 0, y: 0, z: 0 }; // In radians
        this.scale = 1.0;

        // State
        this.enabled = false;
        this.debugMode = false;

        // Cached transformation matrix
        this.transformMatrix = new THREE.Matrix4();
        this._updateMatrix();
    }

    /**
     * Set translation offset
     */
    setTranslation(x, y, z) {
        this.translation.x = x;
        this.translation.y = y;
        this.translation.z = z;
        this._updateMatrix();
    }

    /**
     * Set rotation in radians
     */
    setRotation(x, y, z) {
        this.rotation.x = x;
        this.rotation.y = y;
        this.rotation.z = z;
        this._updateMatrix();
    }

    /**
     * Set rotation in degrees (convenience method)
     */
    setRotationDegrees(x, y, z) {
        this.setRotation(
            x * Math.PI / 180,
            y * Math.PI / 180,
            z * Math.PI / 180
        );
    }

    /**
     * Set uniform scale factor
     */
    setScale(scale) {
        this.scale = scale;
        this._updateMatrix();
    }

    /**
     * Enable or disable transformation
     */
    setEnabled(enabled) {
        this.enabled = enabled;
    }

    /**
     * Enable or disable debug mode
     */
    setDebugMode(enabled) {
        this.debugMode = enabled;
    }

    /**
     * Reset all transformations to identity
     */
    reset() {
        this.translation = { x: 0, y: 0, z: 0 };
        this.rotation = { x: 0, y: 0, z: 0 };
        this.scale = 1.0;
        this._updateMatrix();
    }

    /**
     * Apply transformation to a point (returns new object)
     */
    applyTransform(x, y, z) {
        if (!this.enabled) {
            return { x, y, z };
        }

        const vec = new THREE.Vector3(x, y, z);
        vec.applyMatrix4(this.transformMatrix);

        return {
            x: vec.x,
            y: vec.y,
            z: vec.z
        };
    }

    /**
     * Apply transformation to a THREE.Vector3 (modifies in place)
     */
    applyTransformVector3(vector3) {
        if (!this.enabled) {
            return vector3;
        }

        vector3.applyMatrix4(this.transformMatrix);
        return vector3;
    }

    /**
     * Get the current transformation matrix
     */
    getMatrix() {
        return this.transformMatrix.clone();
    }

    /**
     * Get debug information as formatted string
     */
    getDebugInfo() {
        const m = this.transformMatrix.elements;

        // Format matrix (column-major in Three.js, display as row-major)
        const matrixStr = `┌                                      ┐
│ ${this._formatNum(m[0])}  ${this._formatNum(m[4])}  ${this._formatNum(m[8])}  ${this._formatNum(m[12])} │
│ ${this._formatNum(m[1])}  ${this._formatNum(m[5])}  ${this._formatNum(m[9])}  ${this._formatNum(m[13])} │
│ ${this._formatNum(m[2])}  ${this._formatNum(m[6])}  ${this._formatNum(m[10])}  ${this._formatNum(m[14])} │
│ ${this._formatNum(m[3])}  ${this._formatNum(m[7])}  ${this._formatNum(m[11])}  ${this._formatNum(m[15])} │
└                                      ┘`;

        const paramsStr = `
Translation: (${this.translation.x.toFixed(3)}, ${this.translation.y.toFixed(3)}, ${this.translation.z.toFixed(3)}) m
Rotation: (${(this.rotation.x * 180 / Math.PI).toFixed(1)}°, ${(this.rotation.y * 180 / Math.PI).toFixed(1)}°, ${(this.rotation.z * 180 / Math.PI).toFixed(1)}°)
Scale: ${this.scale.toFixed(3)}`;

        return matrixStr + '\n' + paramsStr;
    }

    /**
     * Update the transformation matrix based on current parameters
     * Applies transformations in order: Scale -> Rotate -> Translate
     */
    _updateMatrix() {
        // Start with identity
        this.transformMatrix.identity();

        // Apply scale
        const scaleMatrix = new THREE.Matrix4().makeScale(
            this.scale,
            this.scale,
            this.scale
        );

        // Apply rotation (Euler angles: X -> Y -> Z order)
        const rotationMatrix = new THREE.Matrix4().makeRotationFromEuler(
            new THREE.Euler(
                this.rotation.x,
                this.rotation.y,
                this.rotation.z,
                'XYZ'
            )
        );

        // Apply translation
        const translationMatrix = new THREE.Matrix4().makeTranslation(
            this.translation.x,
            this.translation.y,
            this.translation.z
        );

        // Combine: Translation * Rotation * Scale
        this.transformMatrix
            .multiply(translationMatrix)
            .multiply(rotationMatrix)
            .multiply(scaleMatrix);
    }

    /**
     * Format number for matrix display
     */
    _formatNum(num) {
        const str = num.toFixed(3);
        return str.padStart(6, ' ');
    }
}
