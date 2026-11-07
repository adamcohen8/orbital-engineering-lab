/* OEL public FSW stdio SDK, protocol oel.fsw.stdio.v1. No third-party dependencies. */
#ifndef OEL_FSW_H
#define OEL_FSW_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#define OEL_FSW_MAX_MESSAGE 262144
/* Views borrow the request and expire when the callback returns.
 * Return zero after writing a NUL-terminated JSON object within capacity;
 * return nonzero for a failed operation. Do not retain borrowed views. */
typedef struct oel_json oel_json;
typedef int (*oel_callback)(const char *operation, const oel_json *payload,
                           char *response, size_t capacity, void *context);
int oel_run(oel_callback callback, void *context, int checkpointable);
int oel_field(const oel_json *json, int object, const char *key);
int oel_find_schema(const oel_json *json, const char *schema);
int oel_vector(const oel_json *json, int object, const char *key, double *values, size_t count);
int oel_raw(const oel_json *json, int token, char *out, size_t capacity);
/* Serialize a typed ideal_wrench.v1 command using the batch's clock and identity. */
int oel_wrench(const oel_json *batch, const char *boot_id_json,
               const double force_n[3], const double torque_n_m[3],
               char *out, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
