# Fase 3 v2 — plan de análisis fijado antes de ver resultados

Fecha de redacción: 9 nov 2026, antes de procesar las imágenes v2 y antes de entrenar ningún modelo v2.
Cualquier cambio posterior a este plan se anota al final, con fecha y motivo.

## 1. Datos y etiquetas

- 16 396 fotos, 160 carpetas (sesiones) = tiempo × clase × genotipo; cosecha 2; 10 genotipos; 4 tiempos.
- DAH = el tiempo de la carpeta (2, 6, 10, 14). Las fechas EXIF se usan solo para identificar el día de captura (8 días = grupo de fotografía × tiempo).
- Clase visual (On-Grade, Minor defects, Major defects, Decay): etiqueta de cada pseudotallo (carpeta); es la única etiqueta individual.
- GQI (cosecha 2, fórmula de la Fase 2) y vida útil restante (t50 en línea de la Fase 1, tope 14 d, menos DAH): etiquetas de celda genotipo × tiempo (40 valores). Se evalúan en la celda.

## 2. Procesamiento de las imágenes (igual para todas las fotos, sin mirar etiquetas)

1. Segmentación de la planta con la función ya publicada (`segment`), control de calidad automático y revisión visual de una muestra.
2. Corrección de color por sesión: ganancia por canal en RGB lineal que lleva la mediana de la mesa (gris oscuro, neutra) de la sesión a la mediana global de la mesa.
3. Recorte solo de la planta; todo lo de fuera de la máscara pasa a gris neutro. Piso, paredes y objetos desaparecen.
4. Alineación: eje mayor horizontal; el extremo más blanco (base del pseudotallo) a la izquierda.
5. Sin referencia de escala: la planta se normaliza a una longitud fija; no se usan tamaños absolutos en píxeles, solo proporciones.
6. Descriptores (≈100): color Lab (media, DE, percentiles), fracciones de color definidas a priori (verde, amarillo verdoso, amarillo, pardo, pálido, oscuro), perfil a lo largo del eje en 10 tramos (anchura relativa, L*, a*, b*), forma sin escala, textura (GLCM, LBP).
7. Embedding ResNet18 (ImageNet, congelada) del recorte alineado de 448 × 112 px, promedio de 4 volteos.

## 3. Modelos (hiperparámetros fijados aquí, no se ajustan con los datos de prueba)

- **M1 · CTD-GB v2**: descriptores → HistGradientBoosting (max_iter 300, learning_rate 0.05, max_depth 4, l2 1.0, min_samples_leaf 40).
- **M2 · CNN-ridge**: embedding ResNet18 estandarizado → Ridge (regresión) / regresión logística multinomial (clases); la regularización se elige por validación cruzada interna agrupada por la unidad del protocolo, solo con datos de entrenamiento.
- **Ensamble** (modelo principal): media simple de M1 y M2 (probabilidades en las clases).
- Entrenamiento con todas las fotos, ponderadas para que cada sesión pese lo mismo.
- **Con DAH conocido**: las mismas variantes con el DAH como entrada adicional (valor añadido de la foto).
- Comparación histórica: CNN-MTL de la versión anterior (foto completa), predicciones ya calculadas.

## 4. Variables respuesta y unidad de evaluación

| Salida | Unidad principal | Otras unidades |
|---|---|---|
| Clase visual | sesión (media de probabilidades) | foto, pseudotallo |
| DAH (edad de la muestra) | celda / muestra | foto, pseudotallo; curva según k pseudotallos |
| GQI | celda (40) | foto |
| Vida útil restante | celda (40) | foto; decisión |

## 5. Protocolos de validación

- P1 sesión (bloque genotipo × tiempo, 5 pliegues, 5 repeticiones con semillas 42–46).
- P1 día de captura (8 pliegues).
- P2 tiempo no visto (4 pliegues).
- **P3 genotipo no visto (10 pliegues) — resultado principal de generalización.**
- P4 grupo de fotografía: entrenar con el grupo A (5 genotipos fotografiados en sus propios días) y probar en el B, y al revés (genotipos y días de foto no vistos).
- Escalera de fuga: foto al azar, pseudotallo, sesión, bloque, día, P2, P3.

Referencias: trivial, solo DAH, solo genotipo, genotipo + DAH.

## 6. Métricas e inferencia

- Regresión: R², MAE; clases: exactitud, kappa ponderado cuadrático, exactitud ±1 clase.
- IC 95 % por bootstrap de conglomerados (2000): bloques en P1, P1 día y P2; genotipos en P3; en P4 no se calcula IC (2 pliegues), se reporta por grupo de fotografía.
- Diferencias pareadas (modelo − referencia) con IC por el mismo bootstrap: es la prueba de si la foto aporta.
- Conformal partido, α = 0.10, calibración agrupada por la unidad del protocolo.
- Decisión: rechazo = vida útil restante ≤ 0; matriz aceptado/rechazado, curvas de costo (FA/FR 1–10), regla por días, decisión con intervalo conformal.

## 7. Controles de atajo

- Comparación de la escalera de fuga v2 frente a la versión con foto completa.
- Oclusión sobre el recorte para ver qué partes de la planta usa M2.
- Validación del color de la cámara (crudo y corregido) frente al colorímetro, por celda.

## 8. Lo que no se hará

No se eligen hiperparámetros, protocolos, genotipos ni umbrales mirando los resultados de prueba. Todo resultado se reporta, también los negativos.

## Cambios posteriores al plan

Todos hechos el 9 nov 2026, **antes** de entrenar cualquier modelo v2 y sin ver ningún resultado de validación.

1. **Segmentación (paso 2.1).** La revisión visual de 44 fotos estratificadas por clase mostró que la función
   publicada `segment` fallaba: con un degradado de luz sobre la mesa dejaba la planta fuera; además metía
   cuñas del borde de la mesa y franjas de pared. Se reemplazó por `segment_v2` (fondo de la mesa modelado con
   una superficie cuadrática, envolvente convexa de la mesa, exclusión de sombras y reflejos, y selección del
   componente principal por color). Se revisó de nuevo en 32 fotos nuevas antes de procesar todas.
   Una primera corrida completa se detuvo en la sesión 28 porque la revisión de 32 recortes al azar y de los
   16 casos extremos mostró (i) plantas muy cercanas a la cámara que parten la mesa en dos (solo se tomaba una
   mitad) y (ii) objetos ajenos (lápiz, regla, ladrillos) sumados como fragmentos. Se corrigió: la mesa es la
   unión de las regiones oscuras y neutras grandes (≥ 15 % de la mayor); los fragmentos y el componente
   principal deben tener color de planta (≥ 30 % de píxeles con C\* ≥ 12 y tono 65–150°); en el 5 % exterior
   de la foto se quitan los píxeles pardos, anaranjados o rojos. Todo se reprocesó desde cero.
   Una segunda corrida se detuvo en la sesión 60: en la muestra al azar apareció **una tarjeta rotulada (fecha y
   genotipo) tomada como planta** y, en otra foto, **una bandeja de cebollas al fondo** sumada a la planta; también
   seguían entrando ladrillos de la pared en el borde de la mesa. Reglas finales: el componente principal se
   pondera por cercanía al centro de la foto; los fragmentos deben medir menos del doble del principal, no ser
   objetos compactos sin color de planta (solidez > 0,85), no tener el centro en el 5 % exterior de la foto ni
   estar sobre el borde de la mesa, y los pequeños deben estar a ≤ 4 % del ancho. Se verificó en los 9 casos
   problemáticos y en 48 fotos al azar estratificadas; se reprocesó todo desde cero.
2. **Orientación (paso 2.4).** «Extremo más blanco» se cambió por «extremo menos verde» (a\* > −6), porque la
   base podrida es parda y no blanca.
3. **Rejillas de regularización de M2**, no fijadas en el plan: Ridge α ∈ 10^[−2, 5] (15 valores), logística
   C ∈ 10^[−5, 0] (6 valores); pesos que igualan las sesiones.
4. **P4 (grupo de fotografía)**: la calibración conformal y la CV interna se separan por genotipo.
5. **Escalera de fuga**: los peldaños bloque, día, P2 y P3 son las mismas predicciones de la validación
   principal; solo foto al azar, pseudotallo y sesión se corren aparte.
6. **Control de calidad de la máscara** (sin etiquetas, antes de validar): una foto cuyo recorte mide menos del
   30 % del largo mediano de la planta se trata como fallo de segmentación y se excluye (1 de 4 123 fotos en la
   revisión de las primeras 40 sesiones). La revisión al azar de 48 recortes de la corrida final mostró 46
   limpios y 2 con un trozo de lápiz junto a la planta (se deja como limitación; es propio de la sesión).
7. **DINOv2** no se usó (requería descargar código externo); M2 usa solo ResNet18, como estaba en el plan.
8. **Terminología** (10 nov, después de los resultados; no cambia ningún análisis): los dos grupos de cinco
   genotipos fotografiados en días distintos, antes «lotes de captura», se llaman **grupos de fotografía**, y P4
   se describe como «genotipos y días de foto no vistos»; el producto evaluado, antes «lote», se llama **muestra**.
9. **Rechazo = vida útil restante < 0** (11 nov, después de los resultados). El plan decía ≤ 0. Con el tope de 14 d,
   ANTSRO029 y ANTSNC003 (t50 en línea 14,1 y 15,4 d) tienen vida útil restante 0 a 14 DAH y quedaban rechazados
   aunque más de la mitad de los consumidores todavía los acepta. Con < 0 se rechaza solo cuando el DAH supera la
   vida útil, la misma definición del GQI (Fase 2). Cambian solo la verdad de la decisión (35,1 % de fotos a rechazar,
   antes 40,2 %), la tabla D12, la Tabla 3b y la columna «Rejected from» de la Tabla 1; los modelos y las
   predicciones no cambian.
